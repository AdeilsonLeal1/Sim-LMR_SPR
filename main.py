import os 
import Reflectivity as ref
import Refractive_index as r_i
import matplotlib.pyplot as plt
import concurrent.futures
import time

from scipy.integrate import *
from numpy import *
from tqdm import tqdm
from scipy.signal import butter, filtfilt


class Simulator_LMRSPR(object):    
    def __init__(self):
        region = {1: 'Same detection region', 2:'Different detection regions'}
        structure = {1:'Kretschmann-Raether configuration', 2:'Optical Fiber configuration', 3:'Planar Waveguide configuration'}
        self.materials = {1: 'BK7', 2:'Silica', 3: 'N-F2', 4: 'Synthetic sapphire(Al2O3)', 5:'SF10', 6: 'FK51A', 7: 'N-SF14', 8:'Acrylic SUVT', 9: 'PVA', 10: 'Glycerin', 11: 'Quartz', 12: 'Gold', 13: 'Silver', 14: 'Copper', 15: 'P3HT:PC61BM', 16:'PEDOT:PSS', 17: '2D HOIP',18:'TiO2', 19:'ZnO', 20:'Water', 21: 'Air', 22:'LiF', 23:'Cytop', 24:'ITO',25:'Analyte', 26: 'WS2', 27: 'Platinum', 28: 'Graphene', 29:'BSTS', 30:'Cobalt', 31:'Nickel', 32:'BaTiO3'}
        
        self.init = 0
        self.final = 0
        self.d = list()         # Thickness
        self.material = list()  # List with the materials of each layer
        self.nLayers = 0        # Amount of layers
        self.d_core = 0         # fiber core diameter
        self.n_aperture = 0     # numerical aperture
        self.list_analyte = list() # List with the  analyte refractive indeces for each variation
        self.analyte = 0        # Initial refractive index (RIU)
        self.step = 0           # Analyte variation step
        self.nvar = 0           # Number of analyte variations
        self.refractive_index = list()     # List with refractive index of each layer
        self.lambda_i = arange(400*1E-9, 1000*1E-9, 1*1E-9)
        self.critical_point = list()    # threshold angle for Attenuated Total Reflection
        self.P_trans_TE = list()
        self.P_trans_TM = list()

        self.lambda_res_TM = list()
        self.lambda_res_TE = list()

        self.sensibility_TM = list()
        self.sensibility_TE = list()

        self.fwhm_TM = list()
        self.fwhm_TE = list()
        self.da_TM = list()
        self.da_TE = list()
        self.qf_TM = list()
        self.qf_TE = list()

        
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

        print(f"{'='*100}\n{' Sensor construction ':=^100}\n{'='*100}\n\n")
        print(f"{'='*100}\n{'Generation of the effects':^100}\n{'='*100}\n")

        reg = int(input("\t1 - In the same detection region;\n\t2 - In different detection regions;\n( 1 or 2 ? )\n=> "))
        print(f"\n-> {region[reg]} has been selected!\n")
    
        print(f"{'='*100}\n{'  Sensor based on what kind of structure  ':^100}\n{'='*100}\n")

        struct = int(input("\t1 - In Kretschmann-Raether configuration;\n\t2 - In Optical Fiber configuration;\n\t3 - In Planar Waveguide configuration;\n( 1, 2 or 3 ? )\n=> "))
        print(f"\n-> {structure[struct]} has been selected!\n")
        
        if reg==1:
            if struct == 1:     # Kretschmann-Raether configuration
                print(f"{'='*100}\n{'Feature in development':^100}\n{'='*100}\n")

            elif struct == 2:   # Optical Fiber configuration
                self.setFiber()
                self.calc_power_optical(2)
                if self.does_it_have_resonance(self.P_trans_TM[0]):
                    self.calc_sensitivity('TM')
                    self.calc_FWHM('TM')
                    self.calc_qf('TM')
                    print(f"Lambda_res_TM: {self.lambda_res_TM}"
                          f"\nSensi_TM:{self.sensibility_TM}"
                          f"\nFWHM_TM:{self.fwhm_TM}"
                          f"\nDA_TM:{self.da_TM}"
                          f"\nQF_TM:{self.qf_TM}")
                else:
                    print(f"{'#'*100}\n{'   There is no resonance in TM polarization   ':^100}\n{'#'*100}\n")
                
                if self.does_it_have_resonance(self.P_trans_TE[0]):
                    self.calc_sensitivity('TE')
                    self.calc_FWHM('TE')
                    self.calc_qf('TE')
                    print(f"Lambda_res_TE: {self.lambda_res_TE}"
                          f"\nSensi_TE:{self.sensibility_TE}"
                          f"\nFWHM_TE:{self.fwhm_TE}"
                          f"\nDA_TE:{self.da_TE}"
                          f"\nQF_TE:{self.qf_TE}")
                else:
                    print(f"{'#'*100}\n{'   There is no resonance in TE polarization   ':^100}\n{'#'*100}\n")
                self.save_file()
                self.plot_curves()
            elif struct == 3:   # Planar Waveguide configuration
                print(f"{'='*100}\n{'Feature in development':^100}\n{'='*100}\n")
                

        if reg==2:
            print(reg)


    def calc_refractive_index(self, lambda_i):
        self.refractive_index = []
        for x in range(len(self.material)):
            if self.material[x] == 25:
                self.refractive_index.append(self.analyte)
            else: 
                self.refractive_index.append(r_i.set_RefractiveIndex(self.material[x], lambda_i))
    
    def setLayers(self):
        # Defines the characteristics of each layer
            
        stringmaterials = self.labels('Materials')
        
        opt = '1'
        while True:
            # It inserts a new layer
            if opt.isnumeric() and int(opt) == 1:
                print(f"\n{f'     Layer {(self.nLayers + 1)}     ':=^100}\n")
                # Assignment of materials
                while True:
                    print(f"{stringmaterials}")
                    material = input(f"\nMaterial -> ")
                    if material.isnumeric() and 0 < int(material) <= len(self.materials):
                        if material == '25':
                            self.setAnalyte()
                            self.material.append(int(material))
                            self.nLayers = len(self.material)
                        else:
                            self.material.append(int(material))
                            self.nLayers = len(self.material)
                        break
                    else:
                        print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
                
                # Assignment of Thickness
                while True:
                    d = float(input("Thickness (nm): "))
                    if d > 0:
                        self.d.append(d*1e-9)
                        break
                    else:
                        print("- {!} - \nInvalid Value!\nPlease enter a positive number...\n")

            # it replicates the previous layer
            elif opt.isnumeric() and int(opt) == 2:
                while True:
                    n = input("How many times do you want to repeat the previous layer?\n -> ")
                    if n.isnumeric() and int(n)>0:
                        for i in range(int(n)):
                            print(f"\n{f'     Layer {(self.nLayers + 1)}     ':=^100}\n")
                            print('-> OK')
                            self.d.append(self.d[-1])
                            self.material.append(self.material[-1])
                            self.nLayers = len(self.material)
                        break
                    else:
                        print("- {!} - \nInvalid Value!\nPlease enter a positive integer...\n")

            # Layer assignment is complete
            elif opt.isnumeric() and int(opt) == 3:
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")

            opt = input(f"{'='*100}\n"
                        "\t1 - New layer \n\t2 - Replicate the previous layer\n"
                        "\t3 - Layer assignments are complete\n op -> ")

        self.nLayers = len(self.material)
        print(f"\n{'='*100}\n{ f'Structure with {self.nLayers}-layers was successfully built':^100}\n{'='*100}\n")

        self.calc_refractive_index(self.lambda_i[0])
        self.labels('Layers')

    def labels(self, label):
        if label == 'Materials':
            stringmaterials = []
            for mat in self.materials:
                stringmaterials.append(f"{f'{mat} - {self.materials[mat]}':<18}")
            
            stringmaterials = f"\t{stringmaterials[0]}{stringmaterials[1]}{stringmaterials[2]}{stringmaterials[3]}\n\t{stringmaterials[4]}{stringmaterials[5]}{stringmaterials[6]}{stringmaterials[7]}\n\t{stringmaterials[8]}{stringmaterials[9]}{stringmaterials[10]}{stringmaterials[11]}\n\t{stringmaterials[12]}{stringmaterials[13]}{stringmaterials[14]}{stringmaterials[15]}\n\t{stringmaterials[16]}{stringmaterials[17]}{stringmaterials[18]}{stringmaterials[19]}\n\t{stringmaterials[20]}{stringmaterials[21]}{stringmaterials[22]}{stringmaterials[23]}\n\t{stringmaterials[24]}{stringmaterials[25]}{stringmaterials[26]}{stringmaterials[27]}\n\t{stringmaterials[28]}{stringmaterials[29]}{stringmaterials[30]}{stringmaterials[31]}"
            
            return stringmaterials
        elif label == 'Layers':
            print(f"\n{'='*100}\n{'   Layers of structure   ':=^100}\n{'='*100}\n")
            print(f"{' '*25}{' Layer':<10} | {'Material':^25} | {'Thickness (nm)':>14}")
            print(f"{' '*25}{'='*57}")
            for i in range(len(self.material)):
                print(f"{' '*25}{f' Layer {i+1}':<10} | {self.materials[self.material[i]]:^25} | {self.d[i]*1E9:>14.2f}")
            print('\n\n')  

    def setFiber(self):
        print(f"{'='*100}\n{'Set Optical Fiber Characteristics':^100}\n{'='*100}\n")

        # Set core diameter
        while True:
            d_core = input(f"Fiber core diameter (D) (\u03BCm): ")
            if d_core.isnumeric() and float(d_core) > 0:
                self.d_core =(float(d_core)*1e-6)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
        # Set numerical aperture
        while True:
            n_aperture = input("Numerical aperture (NA): ")
            n_aperture = float(n_aperture)
            if n_aperture > 0:
                self.n_aperture = float(n_aperture)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
        # Set length of the sensing region
        while True:
            L = input("Length of the sensing region (L) (mm): ")
            
            if float(L) > 0:
                self.L = float(L)*1e-3
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
        
        stringmaterials = self.labels('Materials')
        
        print(f"\n{f'     Core material     ':=^100}\n")
        # Assignment of core material
        while True:
            print(f"{stringmaterials}")
            material = input(f"\nMaterial -> ")
            if material.isnumeric() and 0 < int(material) <= len(self.materials):
                self.material.append(int(material))
                self.d.append(self.d_core)
                self.nLayers = len(self.material)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
        
        self.setLayers()

    def setAnalyte(self):
        print(f"{'='*100}\n{'Set Analyte Characteristics':^100}\n{'='*100}\n")
        while True:
            analyte = input("Initial refractive index (RIU): ")
            analyte = float(analyte)
            if analyte > 0:
                self.analyte = complex(analyte)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid value...\n")
        
        while True:
            step = input("Analyte variation step (\u0394n_a) (RIU): ")
            step = float(step)
            if step > 0:
                self.step = float(step)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a number greater than 0...\n")
        
        while True:
            nvar = input("Number of analyte variations: ")
            nvar = int(nvar)
            if nvar > 1:
                self.nvar = int(nvar)
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a number greater than 1...\n")
        
        self.list_analyte = [self.analyte + self.step*x  for x in range(self.nvar)]
       
    def f_source(self, opc, n_core, wavelenght, theta, W):
        # Equations based on the work of CHIAVAIOLI AND JANNER: Chiavaioli, F., & Janner, D. (2021). Fiber optic sensing with lossy mode resonances: Applications and perspectives. Journal of Lightwave Technology, 39(12), 3855-3870.
        if opc == 1:
            k0 = (2 * pi) / wavelenght
            a = k0*n_core*sin(theta)*cos(theta)
        elif opc == 2:
            x = theta - (pi/2)
            y = 2 * W**2
            a = exp(-(x**2)/y)
        elif opc == 3: 
            num = sin(theta)*cos(theta)
            den = (1 - (n_core**2)*(cos(theta))**2)**2
            a = (n_core**2)*(num/den)
        
        return real(a)
            
    def theta_c(self, na, n_0):
        a = sqrt(1-(na/n_0)**2)
        return arcsin(a)

    def calc_power_optical(self, struct):
        self.init = time.perf_counter()
        print(f"\n{'='*100}\n{f'     Calculating...     ':=^100}\n{'='*100}\n")

        if struct == 1: # Kretschmann-Raether configuration
           pass

        elif struct == 2: # Optical Fiber configuration
            for index_analyte in self.list_analyte:
                P_trans_TM = []
                P_trans_TE = []
                self.critical_point = []
                for pol in ['TM', 'TE']:
                    for t in tqdm(range(len(self.lambda_i))):
                        self.calc_refractive_index(self.lambda_i[t])
                        layer_analyte = self.material.index(25)
                        self.refractive_index[layer_analyte] = index_analyte
                        self.critical_point.append(abs(self.theta_c(self.n_aperture, self.refractive_index[0])) * (180 / pi))
                        
                        f_source = lambda theta3: real((self.refractive_index[0]**2)*((sin(theta3)*cos(theta3))/((1 - (self.refractive_index[0]**2)*(cos(theta3))**2)**2)))
                        
                        #f_source = lambda theta3, W: exp(-((theta3 - (pi/2))**2)/(2 * W**2))

                        r_p = lambda theta, lambda_i: ref.Reflectance(self.nLayers, self.d, self.refractive_index, theta, lambda_i, pol)

                        N = lambda theta2: self.L/(self.d_core*tan(theta2))

                        f = lambda theta4, lambda_i4: f_source(theta4)*(r_p(theta4, lambda_i4)**N(theta4))

                        b = self.lambda_i[t]
                        integral_num = quad(f, self.critical_point[t]*pi/180, pi/2, args=(b))[0]
                        integral_den = quad(f_source, self.critical_point[t]*pi/180, pi/2)[0] 

                        if pol == 'TM':
                            P_trans_tm = integral_num/integral_den
                            P_trans_TM.append(P_trans_tm)
                        else:
                            P_trans_te = integral_num/integral_den
                            P_trans_TE.append(P_trans_te)
                
                #P_trans_TM = list(self.butter_lowpass_filter(P_trans_TM, cutoff=2, fs=40, order=5))
                #P_trans_TE = list(self.butter_lowpass_filter(P_trans_TE, cutoff=2, fs=40, order=5))
                
                self.P_trans_TE.append(P_trans_TE)
                self.P_trans_TM.append(P_trans_TM)
                    
        elif struct == 3: # Planar Waveguide configuration
            with concurrent.futures.ProcessPoolExecutor() as exec:
                self.P_trans = list(exec.map(self.calc_fiber, self.list_analyte))

                for P_trans in self.P_trans:
                    self.P_trans_TM.append(P_trans)

        self.final = time.perf_counter()

    def calc_sensitivity(self, pol):
        lambda_res = []
            
        for n in range(len(self.list_analyte)):
            pot = self.P_trans_TM[n] if pol == 'TM' else self.P_trans_TE[n]
            #max_list, min_list = self.find_max_min( seq = pot)
            idmin = pot.index(min(pot))
            lambda_res = self.lambda_i[idmin]*1E9
           
            #for m in range(len(min_list)):
            #    lambda_res.append(self.lambda_i[min_list[m][0]]*1E9)

            if pol == 'TM':
                self.lambda_res_TM.append(lambda_res)
            else:
                self.lambda_res_TE.append(lambda_res)
        
        #if pol == 'TM':
        #    self.lambda_res_TM = [[row[i] for row in self.lambda_res_TM] for i in range(len(self.lambda_res_TM[0]))]
        #else:
        #    self.lambda_res_TE = [[row[i] for row in self.lambda_res_TE] for i in range(len(self.lambda_res_TE[0]))]
        

        #for n in range(self.n_res):
        sensi_ = []
        delta_lambda_ = []

        for m in range(len(self.list_analyte)):
            if m == 0:
                # The first interaction is initialized to zero because the ratio would be 0/0
                sensi_.append(0)
                delta_lambda_.append(0*1E9)

            else:
                if pol == 'TM':
                    delta_lambda_.append(abs(self.lambda_res_TM[m] - self.lambda_res_TM[m-1]))
                else:
                    delta_lambda_.append(abs(self.lambda_res_TE[m] - self.lambda_res_TE[m-1]))
                
                sensi_.append(round((delta_lambda_[m] / self.step), 6))

        sensi_[0] = sensi_[1]

        if pol == 'TM':
            self.sensibility_TM = sensi_
        else:
            self.sensibility_TE = sensi_
                      
    def calc_fiber(self, index_analyte):
        P_trans_TM = []

        for t in range(len(self.lambda_i)):
            self.calc_refractive_index(self.lambda_i[t])
            layer_analyte = self.material.index(25)
            self.refractive_index[layer_analyte] = index_analyte
            self.critical_point.append(abs(self.theta_c(self.n_aperture, self.refractive_index[0])) * (180 / pi))
            
            f_source = lambda theta3: real((self.refractive_index[0]**2)*((sin(theta3)*cos(theta3))/((1 - (self.refractive_index[0]**2)*(cos(theta3))**2)**2)))
            
            #f_source = lambda theta3, W: exp(-((theta3 - (pi/2))**2)/(2 * W**2))

            
            r_p = lambda theta, lambda_i: ref.Reflectance(self.nLayers, self.d, self.refractive_index, theta, lambda_i)

            N = lambda theta2: self.L/(self.d_core*tan(theta2))

            f = lambda theta4, lambda_i4: f_source(theta4)*(r_p(theta4, lambda_i4)**N(theta4))

            b = self.lambda_i[t]
            integral_num = quad(f, self.critical_point[t]*pi/180, pi/2, args=(b))[0]
            integral_den = quad(f_source, self.critical_point[t]*pi/180, pi/2)[0] 

            P_trans = integral_num/integral_den
            P_trans_TM.append(P_trans)
    
        return P_trans_TM

    def plot_curves(self):
        fig, ax1_TM = plt.subplots(dpi=200)
        l2 = []  # List with the index of refraction of the analyte for plotting the graph
        legend_i = []
        fig2, ax1_TE = plt.subplots(dpi=200)

        while True:
            op = input("Plot in dB?\n\t1-Yes\n\t2-No\n( 1 or 2 ? )\n=> ")
            if op.isnumeric() and int(op)==1: 
                for i in range(self.nvar):
                    ax1_TM.plot(self.lambda_i*1E9, 10*log10(self.P_trans_TM[i]))
                    legend_i.append(fr"{self.list_analyte[i].real:.3f}")
                    l2.append(f"{self.list_analyte[i].real:.3f}")
                ax1_TM.set_title("Trasmitted Optical Power vs. Wavelength - TM", fontsize=12, loc='center', pad='6')
                ax1_TM.set(xlabel=f'Wavelength (nm)', ylabel='Reflectance(dB)')
                ax1_TM.grid(alpha=0.25)
                ax1_TM.legend(legend_i, fontsize=14)

                for i in range(self.nvar):
                    ax1_TE.plot(self.lambda_i*1E9, 10*log10(self.P_trans_TE[i]))
                    legend_i.append(fr"{self.list_analyte[i].real:.3f}")
                    l2.append(f"{self.list_analyte[i].real:.3f}")
                ax1_TE.set_title("Trasmitted Optical Power vs. Wavelength - TE", fontsize=12, loc='center', pad='6')
                ax1_TE.set(xlabel=f'Wavelength (nm)', ylabel='Reflectance(dB)')
                ax1_TE.grid(alpha=0.25)
                ax1_TE.legend(legend_i, fontsize=14)
                plt.show()
                break
            
            elif op.isnumeric() and int(op)==2:
                for i in range(self.nvar):
                    ax1_TM.plot(self.lambda_i*1E9, self.P_trans_TM[i])
                    legend_i.append(fr"{self.list_analyte[i].real:.3f}")
                    l2.append(f"{self.list_analyte[i].real:.3f}")
                ax1_TM.set_title("Reflectance vs. Wavelength - TM", fontsize=12, loc='center', pad='6')
                ax1_TM.set(xlabel=f'Wavelength (nm)', ylabel='Reflectance')
                ax1_TM.grid(alpha=0.25)
                ax1_TM.legend(legend_i, fontsize=14)

                for i in range(self.nvar):
                    ax1_TE.plot(self.lambda_i*1E9, self.P_trans_TE[i])
                    legend_i.append(fr"{self.list_analyte[i].real:.3f}")
                    l2.append(f"{self.list_analyte[i].real:.3f}")
                ax1_TE.set_title("Trasmitted Optical Power vs. Wavelength - TE", fontsize=12, loc='center', pad='6')
                ax1_TE.set(xlabel=f'Wavelength (nm)', ylabel='Reflectance')
                ax1_TE.grid(alpha=0.25)
                ax1_TE.legend(legend_i, fontsize=14)
                plt.show()
                break
            else:
                print("- {!} - \nInvalid Value!\nPlease enter a valid option...\n")
        
    def find_max_min(self, seq):
        max_list = []
        min_list = []

        for i in range(1, len(seq) - 1):
            if i == 1 and seq[i] > seq[i+1]: 
                max_list.append((i, seq[i]))
            elif i == (len(seq) - 2) and seq[i] >= seq[i-1]:
                max_list.append((i, seq[i]))
            elif seq[i] > seq[i-1] and seq[i] > seq[i+1]:
                max_list.append((i, seq[i]))
            elif seq[i] < seq[i-1] and seq[i] < seq[i+1]:
                min_list.append((i, seq[i]))
        
        return max_list, min_list
        
    def calc_FWHM(self, pol):
        fwhm_ = []
        da_ = []        
        for n in range(len(self.list_analyte)):

            curve = self.P_trans_TM[n] if pol == 'TM' else self.P_trans_TE[n]
            #max_list, min_list = self.find_max_min(curve)
            #for m in range(len(max_list)-1):
            #lambda_i = list(self.lambda_i[max_list[m][0]:max_list[m+1][0]])
            #y = list(curve[max_list[m][0]:max_list[m+1][0]])
            y = curve

            id_min = y.index(min(y))  # Position of the minimum point of the curve

            y_left = y[0:(id_min+1)]
            y_right = y[id_min:len(y)]

            y_mx_left = max(y_left)
            y_mn_left = min(y_left)

            y_mx_right = max(y_right)
            y_mn_right = min(y_right)

            y_med_left = (y_mx_left + y_mn_left)/2
            y_med_right = (y_mx_right + y_mn_right)/2

            y_med = (y_med_left + y_med_right)/2
            #y_med = (1+min(y))/2    
            #y_med = (max(y) + min(y))/2 
            #y_med = max(y)/2
            #y_med = y_mx_right
            
            try:
                # Gaur's methodology for calculating FWHM
                y_left = asarray(y_left)
                y_right = asarray(y_right)
                idx1 = (abs(y_left - y_med_left)).argmin() 
                idx2 = (abs(y_right - y_med_right)).argmin() 
                x1= self.lambda_i[idx1]*1E9
                x2= self.lambda_i[id_min + idx2]*1E9
                
                f = sqrt(abs((x2-x1))**2 + abs(y_med_right - y_med_left)**2)
                
                fwhm_.append(round(f, 6))
                da_.append(round(1/f, 6))
            except:
                fwhm_.append(1)
                da_.append(1)

            if pol == 'TM':
                self.fwhm_TM = fwhm_
                self.da_TM = da_
            else:
                self.fwhm_TE = fwhm_
                self.da_TE = da_

        #if pol == 'TM':
        #    self.fwhm_TM = [[row[i] for row in self.fwhm_TM] for i in range(len(self.fwhm_TM[0]))]
        #    self.da_TM = [[row[i] for row in self.da_TM] for i in range(len(self.da_TM[0]))]
        #else:
        #    self.fwhm_TE = [[row[i] for row in self.fwhm_TE] for i in range(len(self.fwhm_TE[0]))]
        #    self.da_TE = [[row[i] for row in self.da_TE] for i in range(len(self.da_TE[0]))]

    def calc_qf(self, pol):
        #for n in range(self.n_res):
        qf_ = []
        for m in range(len(self.list_analyte)):
                if pol == 'TM':
                    qf_.append(round(self.sensibility_TM[m]/self.fwhm_TM[m],6))
                else:
                    qf_.append(round(self.sensibility_TE[m]/self.fwhm_TE[m],6))

        if pol == 'TM':
            self.qf_TM.append(qf_)
        else:
            self.qf_TE.append(qf_)
  
    def does_it_have_resonance(self, pot):
        max_list, min_list = self.find_max_min(pot)
        if len(max_list)<2:
            return False
        else:
            self.n_res = len(max_list)-1
            return True

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        # Get the filter coefficients 
        b, a = butter(order, cutoff, btype='low', fs = fs)
        y = filtfilt(b, a, data)
        return y

    def save_file(self):
        mode_p = ["TM", "TE"]

        for mode in mode_p:
            for i in range(len(self.list_analyte)):
                file_reflectance = open(f'Simulação_Power_transmitted_vs_analyte_{str(real(self.list_analyte[i])).replace(".","_")}_{mode}.txt', 'w')
                file_reflectance.write(f"Wavelength,Power transmitted")
                if mode == "TE":
                    for k in range(len(self.lambda_i)):
                        file_reflectance.write(f"\n{self.lambda_i[k]*1E9:.3f},{self.P_trans_TE[i][k]:.6f}")
                else:
                    for k in range(len(self.lambda_i)):
                        file_reflectance.write(f"\n{self.lambda_i[k]*1E9:.3f},{self.P_trans_TM[i][k]:.6f}")
                file_reflectance.close()

if __name__ == "__main__":
    while True:
        print(f"{'='*100}\n{'   Welcome to Sim-LMR+SPR   ':=^100}\n{'='*100}\n")

        op = input("Start a new simulation?\n\t1-Yes\n\t2-No\n( 1 or 2 ? )\n=> ")
        if op.isnumeric() and int(op)==1:
            sim = Simulator_LMRSPR()
        elif op.isnumeric() and int(op)==2:
            break
        else:
            print("- {!} - \nInvalid Value!\nPlease enter a valid option...\n")

