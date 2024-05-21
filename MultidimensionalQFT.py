import numpy as np
import qulacs
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable 

class Preprocessing():
    def __init__(self, coefficients_scattering_factor: np.ndarray)->None:
        self.clear(coefficients_scattering_factor)
        
    def clear(self, coefficients_scattering_factor: np.ndarray)->None:
        self.coefficients_scattering_factor = coefficients_scattering_factor
        self.coefficients_window            = np.array([0.1881,0.36923,0.28702,0.13077,0.02488])
    
    def preprocessing_parser(self, XFTinput: np.ndarray, preprocessing_order: dict)->None:
        for preprocessing in preprocessing_order["preprocessings"]:
            if preprocessing =="tile":
                XFTinput = self.tile(XFTinput,preprocessing_order["tile_number"])
                if preprocessing_order.get("d").all():
                    preprocessing_order["d"]=preprocessing_order["d"]*preprocessing_order["tile_number"]
            elif preprocessing =="cutoff":
                XFTinput = self.cutoff(XFTinput,preprocessing_order["cutoff_length"])
            elif preprocessing =="padding":
                if preprocessing_order["padding_values"] == None:
                    XFTinput = self.padding(XFTinput,preprocessing_order["cutoff_length"])
                else:
                    XFTinput = self.padding(XFTinput,preprocessing_order["cutoff_length"],preprocessing_order["padding_values"])
            elif preprocessing =="multiple_flat_top_window":
                XFTinput = self.multiple_flat_top_window(XFTinput)
            elif preprocessing =="multiple_scattering_factor":
                XFTinput = self.multiple_scattering_factor(XFTinput,preprocessing_order["d"],preprocessing_order["DW"])
            elif preprocessing =="normalize":
                XFTinput = self.normalize(XFTinput)
        return XFTinput
        
    def tile(self, XFTinput: np.ndarray, tile_number: np.ndarray)->np.ndarray:
        return np.tile(XFTinput, tuple(tile_number))
        
    def cutoff(self, XFTinput: np.ndarray, cutoff_length: np.ndarray)->np.ndarray:
        if len(cutoff_length) != len(XFTinput.shape):
            raise ValueError(f"The dimension between XFTinput({len(XFTinput.shape)}) and cutoff_length({len(cutoff_length)}) are not matched")
        if len(cutoff_length) == 1:
            return XFTinput[:cutoff_length[0]]
        elif len(cutoff_length) ==2:
            return XFTinput[:cutoff_length[0],:cutoff_length[1]]
        else:
            raise ValueError(f"The dimension {len(cutoff_length)} >= 3 is not implemented, however you can implement to cutoff fucntion @ Preprocessing class.")

    def padding(self, XFTinput: np.ndarray, cutoff_length: list, padding_values=0.0)->np.ndarray:
        if len(cutoff_length) != len(XFTinput.shape):
            raise ValueError(f"The dimension between XFTinput({len(XFTinput.shape)}) and cutoff_length({len(cutoff_length)}) are not matched")
        if isinstance(padding_values, (int, float, complex)): # padding_values can take the type of int, float, complex
            padded_XFTinput =  np.zeros(cutoff_length)
            padded_XFTinput += padding_values
        elif isinstance(padding_values, np.ndarray):          # padding_values can also take the type of np.ndarray
            padded_XFTinput = padding_values
        if len(cutoff_length) == 1:
            padded_XFTinput[:XFTinput.shape[0]] += XFTinput
            return padded_XFTinput
        elif len(cutoff_length) ==2:
            padded_XFTinput[:XFTinput.shape[0],:XFTinput.shape[1]] += XFTinput
            return padded_XFTinput
        else:
            raise ValueError(f"The dimension {len(cutoff_length)} >= 3 is not implemented, however you can implement to padding fucntion @ Preprocessing class.")

    def multiple_flat_top_window(self, XFTinput: np.ndarray)->np.ndarray:
        x_grid,y_grid = np.meshgrid(np.arange(XFTinput.shape[1]),np.arange(XFTinput.shape[0]))
        x_window      = np.sum(((-1)**np.arange(5)*self.coefficients_window)[np.newaxis,np.newaxis,:]*np.cos(2*np.pi/XFTinput.shape[1]*np.arange(5)*np.repeat(x_grid[:,:,np.newaxis],5,axis=2)),axis=2)
        y_window      = np.sum(((-1)**np.arange(5)*self.coefficients_window)[np.newaxis,np.newaxis,:]*np.cos(2*np.pi/XFTinput.shape[0]*np.arange(5)*np.repeat(y_grid[:,:,np.newaxis],5,axis=2)),axis=2)
        self.window   = y_window * x_window
        return XFTinput*self.window
    
    def multiple_scattering_factor(self, XFTinput:np.ndarray, d:np.ndarray, DW:"\sigma^2 of float")->np.ndarray: 
        x_grid,y_grid     = np.meshgrid(np.arange(XFTinput.shape[1])/(d[1]*XFTinput.shape[1]),np.arange(XFTinput.shape[0])/(d[0]*XFTinput.shape[0]))
        s_grid            = np.sqrt(x_grid**2+y_grid**2)
        scattering_factor = np.sum(self.coefficients_scattering_factor[0][np.newaxis,np.newaxis,:]*np.exp(-(self.coefficients_scattering_factor)[1][np.newaxis,np.newaxis,:]+DW)*(np.repeat(s_grid[:,:,np.newaxis],5,axis=2)**2),axis=2)
        return XFTinput*scattering_factor
        
    def normalize(self, XFTinput: np.ndarray)->np.ndarray:
        return XFTinput/np.sqrt(np.sum(XFTinput**2))
    
class FTManager():
    # setting
    def __init__(self):
        self.clear_circuit()

    def __call__(self,XFTinput_list:list[np.ndarray],save_path=None)->list[np.ndarray]: # 2D-DFT, FFT, QFT            まだ
        DFTinput,FFTinput,QFTinput = XFTinput_list
        DFT_F                      = self.DFT_2D(DFTinput)
        FFT_F                      = self.FFT_2D(FFTinput)
        Qubits, bit_division       = self.makeQubit(QFTinput)
        QFT_F_state                = self.NDQFT_qulacs(Qubits,bit_division)
        QFT_F                      = self.rearrange_QFT_results_qulacs(QFT_F_state,bit_division)
        self.draw_heatmap(DFT_F,FFT_F,QFT_F,save_path)
        return [DFT_F,FFT_F,QFT_F]

    def clear_circuit(self):
        self.circuit = None
        
    def makeQubit(self, QFTinput:np.ndarray)->tuple[qulacs.QuantumState,list]:
        bit_division = np.log2(QFTinput.shape).astype(np.int32)
        flat_input   = QFTinput.reshape(-1)
        state        = QuantumState(np.log2(len(flat_input)).astype(np.int16))
        state.load(flat_input)
        return state, bit_division
    
    # Classical FTs
    def DFT_2D(self, DFTinput:np.ndarray, bit_division=None, inverse=False)->np.ndarray:
        np.set_printoptions(precision=8,suppress=True)
        if bit_division is None:
            bit_division = np.log2(DFTinput.shape).astype(np.int16)
        else:
            DFTinput = DFTinput.reshape(list(2**np.array(bit_division)))
        f_xy = DFTinput
        F_s  = []
        F_ts = []
        for x in range(2**bit_division[0]):
            F_s.append(list(self.DFT_1D(f_xy[x,:],inverse)))
        F_s = np.array(F_s)
        for y in range(2**bit_division[1]):
            F_ts.append(self.DFT_1D(F_s[:,y],inverse))
        F_st = np.array(F_ts).T
        return F_st

    def FFT_2D(self, FFTinput:np.ndarray, bit_division=None, inverse=False, test=False)->np.ndarray:
        np.set_printoptions(precision=8,suppress=True)
        if bit_division is None:
            bit_division = np.log2(FFTinput.shape).astype(np.int16)
        else:
            FFTinput = FFTinput.reshape(list(2**np.array(bit_division)))
        f_xy = FFTinput
        F_s  = []
        F_ts = []
        for x in range(2**bit_division[0]):
            F_s.append(list(self.FFT_1D(f_xy[x,:],inverse)))
        F_s = np.array(F_s)
        for y in range(2**bit_division[1]):
            F_ts.append(self.FFT_1D(F_s[:,y],inverse))
        F_st = np.array(F_ts).T
        if test:
            F_st = np.fft.fft2(f_xy)
        return F_st
    
    def DFT_1D(self, DFTinput:np.ndarray, inverse=False)->np.ndarray:
        np.set_printoptions(precision=8,suppress=True)
        f_x = DFTinput
        N   = len(DFTinput)
        x   = np.arange(N) 
        t   = x # len(t) = len(x)
        if inverse:
            F_t = (np.sum(f_x[:,np.newaxis]*np.exp(1j*2*np.pi*x[:,np.newaxis]*t[np.newaxis,:]/N),axis=0))/np.sqrt(N) # [x,t]->[t]
        else:
            F_t = (np.sum(f_x[:,np.newaxis]*np.exp(-1j*2*np.pi*x[:,np.newaxis]*t[np.newaxis,:]/N),axis=0))/np.sqrt(N) # [x,t]->[t]
        return F_t

    def FFT_1D(self, FFTinput:np.ndarray, inverse=False)->np.ndarray:
        f_x = FFTinput
        N   = len(FFTinput)
        if inverse:
            F_t = np.fft.fft(f_x)/np.sqrt(N)
        else:
            F_t = np.fft.ifft(f_x)*np.sqrt(N)
        return F_t
    
    # Quntum FT
    def Ri_gate_qulacs(self, param:int, target:int, control:int,inverse=False)->qulacs.gate.DenseMatrix:
        if inverse:
            gate = DenseMatrix(target, [[1,0],[0,np.exp(-(0 + 1j) *2 * np.pi /2**(param))]])
        else:
            gate = DenseMatrix(target, [[1,0],[0,np.exp((0 + 1j) *2 * np.pi /2**(param))]]) 
        gate.add_control_qubit(control, 1)
        return gate

    def NDQFT_qulacs(self, qubits:qulacs.QuantumState, bit_division:list,inverse=False)->qulacs.QuantumState:
        bit_size = qubits.get_qubit_count()
        if self.circuit is None:
            self.circuit = QuantumCircuit(bit_size)
            target_bit = bit_size-1
            for dimension in range(len(bit_division)):
                for target_difference in range(bit_division[dimension]):
                    target_index = target_bit-target_difference
                    self.circuit.add_H_gate(target_index)
                    R_lim = target_bit-bit_division[dimension]+1
                    for control_index in range(target_index-1,R_lim-1,-1):
                        self.circuit.add_gate(self.Ri_gate_qulacs(target_index-control_index+1, target_index, control_index, inverse))
                target_bit -= bit_division[dimension]
        self.circuit.update_quantum_state(qubits)
        return qubits
    
    def rearrange_QFT_results_qulacs(self, Qubits:qulacs.QuantumState, bit_division:list)->np.ndarray:
        bit_size = Qubits.get_qubit_count()
        raw_state = Qubits.get_vector()
        transpose_index = []
        axis_count      = 0
        for bit_length in bit_division:
            transpose_index += list(np.arange(bit_length-1,-1,-1)+axis_count)
            axis_count += bit_length
        state = raw_state.reshape(2*np.ones(bit_size,np.int16)).transpose(transpose_index).reshape(2**np.array(bit_division))
        return state
    
    # create figures
    def draw_heatmap(self, DFT_F:np.ndarray, FFT_F:np.ndarray, QFT_F:np.ndarray, save_path:str)->None:
        fontsize = 20
        for name,spectrum in zip(["DFT","FFT","QFT"],[DFT_F,FFT_F,QFT_F]):
            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.tick_params(bottom=False, left=False, right=False, top=False)
            plt.xticks([0, len(spectrum[0])-1], [str(0), str(len(spectrum[0])-1)])
            plt.yticks([0, len(spectrum[1])-1], [str(0), str(len(spectrum[1])-1)])
            image   = ax.imshow(np.abs(spectrum).T, origin='lower', cmap="viridis")
            xs, ys  = np.meshgrid(range(spectrum.real.shape[0]),range(spectrum.real.shape[1]),indexing='ij')
            # if you show the data value, delete next and the next head "#" and change fontsize  
            #for x,y,val in zip(xs.reshape(-1), ys.reshape(-1), np.abs(spectrum).reshape(-1)):
                #ax.text(x,y,'{0:.2f}'.format(val), horizontalalignment='center',verticalalignment='center',fontsize=4, path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white', capstyle="round")])
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right",size="5%",pad=0.1)
            cbar    = plt.colorbar(image, cax=cax)
            cbar.set_label('intensity', size=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            ax.set_xlabel("x", fontsize=fontsize)
            ax.set_ylabel("y", fontsize=fontsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if save_path is not None:
                plt.savefig(save_path+f"/{name}.svg",bbox_inches='tight')
            else:
                plt.show()
            plt.clf()
            plt.close()