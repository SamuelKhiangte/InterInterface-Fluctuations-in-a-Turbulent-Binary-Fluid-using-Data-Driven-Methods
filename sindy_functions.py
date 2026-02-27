import numpy as np
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.differentiation import FiniteDifference
from sindy_utils import *
from scipy.integrate import odeint
from scipy.linalg import eig
import struct

import os
import matplotlib.pyplot as plt
from skimage import feature
import struct

import os

from skimage import feature
import re
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import mpld3


def deal_with_periodic(reconstructed_x_coords,reconstructed_y_coords,N):
    Lx = 1024
    Ly = 1024
    
    center_x = Lx / 2
    center_y = Ly / 2
    
    x_all_list=[]
    y_all_list=[]
    
    for n in range(len(reconstructed_x_coords)):  # Loop over all possible values of n
        
        x_list_shift = reconstructed_x_coords[n]
        y_list_shift = reconstructed_y_coords[n]
    
        
        x_list_toshift,y_list_toshift=shift_coordinates_periodically(x_list_shift,y_list_shift,Lx,Ly)
        #plt.scatter(x_list_toshift,y_list_toshift)
        x_coords=x_list_toshift
        y_coords=y_list_toshift
        import numpy as np
        import matplotlib.pyplot as plt
    
    
    
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
    
        
        points = np.array(list(zip(x_coords, y_coords)))
        
        if len(points)>=1:
        
            distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
            #print(distances)
            angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
    
       
            discretized_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
            max_radius_points = []
            min_radius_points = []
            filtered_points = []
            prev_point = None
    
            for angle in discretized_angles:
                tolerance = (np.pi / N) + 0.005
                mask = (angles >= angle - tolerance) & (angles < angle + tolerance) 
        
                if np.any(mask):
                    selected_points = points[mask]
                    selected_distances = distances[mask]
                    max_distance_index = np.argmax(selected_distances)
                    min_distance_index = np.argmin(selected_distances)
                
                    filtered_points.append(selected_points[max_distance_index])
                    max_radius_points.append(selected_points[max_distance_index])
                    min_radius_points.append(selected_points[min_distance_index])
            
                    prev_point = selected_points[max_distance_index]
                else:
                    if prev_point is not None:
                        filtered_points.append(prev_point)
                        max_radius_points.append(prev_point)
                        min_radius_points.append(prev_point)
    
            max_radius_points = [tuple(point) for point in max_radius_points]
            min_radius_points = [tuple(point) for point in min_radius_points]
            filtered_points = [tuple(point) for point in filtered_points]
    
            filtered_x_coords = [point[0] for point in filtered_points]
            filtered_y_coords = [point[1] for point in filtered_points]
    
            x_list_new = filtered_x_coords
            y_list_new = filtered_y_coords
    
        
            #print(len(max_radius_points))
    
            x_all_list.append( x_list_new)
            y_all_list.append( y_list_new)
	return x_all_list,y_all_list

def find_model(u_sub,thetas,ti,r):
    
    tf=ti+3000
    u=np.array(u_sub).T[:,ti:tf]
    t=np.arange(ti,tf,1)

    tot_t_len=len(u[0])
    ip=inner_product(u,thetas)
    x, feature_names, S2, Vh=vector_POD(ip,t,5)
    t_test, x_true,x_train, x_sim, S2,coeffecients, feature_names = compressible_Framework(ip,t,0.04, r=r,tfac=0.95)
    print('Done with SINDy')
    
    Q=u
    # Define portions of the data for testing/training
    M_test = len(t_test)
    M_train = int(len(t) * tfac)
    t_train = t[:M_train]
    # Q_test = Q[:, M_train:]
    Vh_true = np.transpose(x_true)
    Vh_sim_unscaled = np.transpose(x_sim)
    Vh_sim=(np.max(Vh_true)*Vh_sim_unscaled)/np.max(Vh_sim_unscaled)
    Sr = np.sqrt(S2[0:r, 0:r])
    w, v = eig(ip)
    Vh = np.transpose(v).real
    wr = np.sqrt(np.diag(w))
    
    # Compute the POD and SINDy reconstructions of Q
    U = Q@(np.transpose(Vh)[:,:2 * r]@(np.linalg.inv(wr)[:2 * r, :r]).real)  
    U_true = U[:, 0:r]
    
    Q_sim = (U_true @ wr[0:r, 0:r] @ Vh_sim).real
    Q_real = (U_true @ wr[0:r, 0:r] @ Vh_true).real
    
    
    print('Done with POD, SINDy, and DMD, now simulating')
 
   
    perimeter_data=[]
    perimeter_predict=[]
    thetas = np.linspace(0, 2 * np.pi, len(normalized_h_store[0]))
    for i in range(0,len(t_test)):
        sim=Q_sim
        real=Q_real
        
        max_sim = max(max(h) for h in sim)
        max_real = max(max(h) for h in real)
    
        plt.plot(thetas, sim[:,i]/ max_sim, color='r',linewidth=4)
        plt.plot(thetas, real[:,i]/ max_real, color='black',linewidth=2)
    
        perimeter_data.append(calculate_perimeter(sim[:,i]/ max_sim,thetas))
        perimeter_predict.append(calculate_perimeter(real[:,i]/ max_real,thetas))
    return  perimeter_data, perimeter_predict,coeffecients, feature_names



def plot_x_fits(x_test, t_test, optimizer, n_models,r,model):
    pod_names = ["a{}".format(i) for i in range(1, r + 1)]
    model_x_store=[]
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(r, n_models)
    gs.update(wspace=0.0, hspace=0.0) 
    for j in range(n_models):
        optimizer.coef_ = np.asarray(optimizer.history_)[j, :, :]

        # Simulate dynamic system with the identified model
        # Some of these models may diverge, so need to use odeint 
        # (which just gives a warning)
        # rather than the default solve_ivp, which crashes with an error.
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator='odeint')
        model_x_store.append( x_test_sim)
        for i in range(r):
            plt.subplot(gs[i, j])
            plt.plot(t_test, x_test[:, i]/np.max(x_test[:, i]), 'k', label='test trajectory')
            if np.max(np.abs(x_test_sim[:, i])) < 1000:  # check for instability
                plt.plot(t_test, x_test_sim[:, i]/np.max(x_test_sim[:, i]), 'r', label='model prediction')
            if j == 0:
                plt.ylabel(pod_names[i], fontsize=20)
            if i == 0:
                plt.title('MSE = %.0f' % model.score(x_test, 
                                                     t=dt, 
                                                     metric=mean_squared_error), 
                          fontsize=16)
            plt.xlabel('Iteration ' + str(j), fontsize=16)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.grid(True)
   
        print('Model ' + str(j) + ', MSE: %f' % model.score(x_test, 
                                                            t=dt, 
                                                            metric=mean_squared_error))
    ax.set_yticklabels([])
    plt.show()
    return  model_x_store

def compressible_Framework(inner_prod, time, threshold,r, tfac):

    # Define a portion of the data for training and for testing
    M_train = int(len(time) * tfac)
    t_train = time[:M_train]
    t_test = time[M_train:]

    # Compute the dimensionalize POD to get the temporal POD modes in x
    x, feature_names, S2, Vh, = vector_POD(inner_prod, time, r)
   
    
   
    # split the temporal POD modes into testing and training
    x_train = x[:M_train, :]
    x_test = x[M_train:, :]
    x0_train = x[0, :]
    x_true = x[M_train:, :]
    x0_test = x[M_train, :]

    polynomial_library = ps.PolynomialLibrary(degree=2)
    
    optimizer = ps.FROLS(kappa=1e-9,alpha=1e-9, normalize_columns=True)

    ssr_optimizer2 = ps.SSR(criteria="model_residual")
    #model = ps.SINDy(optimizer=ssr_optimizer2 )
    model = ps.SINDy(feature_library= polynomial_library , optimizer=ssr_optimizer2 )
    model.fit(x_train, t=dt)
    model.print()
    coeff=model.coefficients()
    feature_names = model.get_feature_names()

    # Repeat plots but now integrate the ODE and compare the test trajectories
    x_sim_list=plot_x_fits(x_test, t_test, ssr_optimizer2,3,r,model)
    x_sim=x_sim_list[-1]

    
   

    # Rescale back to original units
    for i in range(r):
        x_sim[:, i] = x_sim[:, i] * sum(np.amax(abs(Vh), axis=1)[0:r])
        x_true[:, i] = x_true[:, i] * sum(np.amax(abs(Vh), axis=1)[0:r])

    return t_test, x_true,x_train, x_sim, S2,   coeff, feature_names




def plot_x_dot_fits(x_test, optimizer, dt, n_models,r):
    model_x_dot_store=[]
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(r, n_models)
    gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
    for j in range(n_models):
        optimizer.coef_ = np.asarray(optimizer.history_)[j, :, :]

        # Predict derivatives using the learned model
        x_dot_test_predicted = model.predict(x_test)  
        model_x_dot_store.append(x_dot_test_predicted)
        # Compute derivatives with a finite difference method, for comparison
        x_dot_test_computed = model.differentiate(x_test, t=dt)

        for i in range(r):
            plt.subplot(gs[i, j])
            plt.plot(t_test, x_dot_test_computed[:, i],
                        'k', label='numerical derivative')
            plt.plot(t_test, x_dot_test_predicted[:, i],
                        'r', label='model prediction')
            if j == 0:
                plt.ylabel('$\dot ' + pod_names[i] + r'$', fontsize=20)
            if i == 0:
                plt.title('MSE = %.0f' % model.score(x_test, 
                                                     t=dt, 
                                                     metric=mean_squared_error), 
                          fontsize=16)
            plt.xlabel('Iteration ' + str(j), fontsize=16)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.grid(True)

        model.print()     
        print('Model ' + str(j) + ', MSE: %f' % model.score(x_test, 
                                                            t=dt, 
                                                            metric=mean_squared_error))
    ax.set_yticklabels([])
    plt.show()
    return  model_x_dot_store

    
def plot_x_fits(x_test, t_test, optimizer, n_models,r):
    model_x_store=[]
    plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(r, n_models)
    gs.update(wspace=0.0, hspace=0.0) 
    for j in range(n_models):
        optimizer.coef_ = np.asarray(optimizer.history_)[j, :, :]

        # Simulate dynamic system with the identified model
        # Some of these models may diverge, so need to use odeint 
        # (which just gives a warning)
        # rather than the default solve_ivp, which crashes with an error.
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator='odeint')
        model_x_store.append( x_test_sim)
        for i in range(r):
            plt.subplot(gs[i, j])
            plt.plot(t_test, x_test[:, i], 'k', label='test trajectory')
            if np.max(np.abs(x_test_sim[:, i])) < 1000:  # check for instability
                plt.plot(t_test, x_test_sim[:, i], 'r', label='model prediction')
            if j == 0:
                plt.ylabel(pod_names[i], fontsize=20)
            if i == 0:
                plt.title('MSE = %.0f' % model.score(x_test, 
                                                     t=dt, 
                                                     metric=mean_squared_error), 
                          fontsize=16)
            plt.xlabel('Iteration ' + str(j), fontsize=16)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.grid(True)
   
        print('Model ' + str(j) + ', MSE: %f' % model.score(x_test, 
                                                            t=dt, 
                                                            metric=mean_squared_error))
    ax.set_yticklabels([])
    plt.show()
    return  model_x_store


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


##### Calculate Perimeter ###################
def calculate_perimeter(radius, angles):
    
    # Ensure the inputs are numpy arrays
    radius = np.array(radius)
    angles = np.array(angles)
    
    # Calculate the differences in radius and angles
    delta_r = np.diff(radius)
    delta_theta = np.diff(angles)
    
    # Calculate the incremental distances
    ds = np.sqrt(delta_r**2 + (radius[:-1] * delta_theta)**2)
    
    # Sum the incremental distances to get the perimeter
    perimeter = np.sum(ds)
    
    return perimeter

def format_func(value, tick_number):
    N = int(np.round(value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi$"
    elif N == -1:
        return r"-$\pi$"
    else:
        return r"${0}\pi$".format(N)
def read_binary_file(file_path):
    data = []
    try:
        with open(file_path, 'rb') as file:
            while True:
                bytes_read = file.read(4) # Assuming each number is a 4-byte float
                if not bytes_read:
                    break
                number = struct.unpack('f', bytes_read)[0]
                data.append(number)
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def save_coords_to_file(file_path, x_coords, y_coords):
    try:
        with open(file_path, 'w') as file:
            for x, y in zip(x_coords, y_coords):
                file.write(f"{x} {y}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

def read_coords_from_file(file_path):
    x_coords = []
    y_coords = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split())
                x_coords.append(x)
                y_coords.append(y)
    except Exception as e:
        print(f"Error reading file: {e}")
    return np.array(x_coords), np.array(y_coords)

def process_files(input_folder, output_folder,output_folder2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = sorted([f for f in os.listdir(input_folder) if f.startswith("phi.")])
    x_coords_over_time = []
    y_coords_over_time = []
    #print(files)
    print(len(files))
    for ind,filename in enumerate(files[10074:]):
        ind=ind+10074
        # Extract integer value from the filename using regular expression
        
        
        file_path = os.path.join(input_folder, filename)
        numerical_data = read_binary_file(file_path)
        # Ensure the data length matches the required size for reshaping
        required_size = 1024 * 1024 * 2
        if len(numerical_data) != required_size:
            raise ValueError(f"Data size mismatch in file {filename}. Expected {required_size} elements, but got {len(numerical_data)}.")
        # Reshape the data into a 1024 x 1024 x 2 matrix
        matrix = np.array(numerical_data).reshape((1024, 1024, 2))
        # Perform edge detection on matrix[:, :, 1]
        edges = feature.canny(matrix[:, :, 1])
        # Get the coordinates of the edges
        y_coords, x_coords = np.where(edges)
        x_coords_over_time.append(x_coords)
        y_coords_over_time.append(y_coords)
        # Save the coordinates to a text file
        coords_file_path = os.path.join(output_folder2, filename.replace("phi.", "coords_") + ".txt")
        save_coords_to_file(coords_file_path, x_coords, y_coords)
        # Save the image of the edges
        plt.imshow(edges, cmap='gray')
        #if ind%10==0:
        #    output_file_path = os.path.join(output_folder, filename.replace("phi.", "image_") + ".png")
        #    plt.savefig(output_file_path)
        #    plt.close()
    return x_coords_over_time, y_coords_over_time


def reconstruct_arrays_from_files(output_folder):
    x_coords_over_time = []
    y_coords_over_time = []
    files = sorted([f for f in os.listdir(output_folder) if f.startswith("coords_") and f.endswith(".txt")])
    for filename in files:
        file_path = os.path.join(output_folder, filename)
        x_coords, y_coords = read_coords_from_file(file_path)
        x_coords_over_time.append(x_coords)
        y_coords_over_time.append(y_coords)
    return x_coords_over_time, y_coords_over_time
    
    
    
def shift_coordinates_periodically(x_list, y_list, Lx, Ly):
    # Initialize lists for shifted coordinates
    x_list_shift = []
    y_list_shift = []
    shitf_hand=0
    for x, y in zip(x_list, y_list):
        # Shift x coordinate if difference is greater than half the box size Lx
        
        if abs(x - x_list[0]) > Lx/2:
           
            if x > x_list[0]:
                x -= Lx+shitf_hand
            else:
                x += Lx+shitf_hand

        # Shift y coordinate if difference is greater than half the box size Ly
        if abs(y - y_list[0]) > Ly/2:
            if y > y_list[0]:
                y -= Ly+shitf_hand
            else:
                y += Ly+shitf_hand

        # Append shifted coordinates to the new lists
        x_list_shift.append(x)
        y_list_shift.append(y)

    return x_list_shift, y_list_shift



def apply_periodic_boundary(value, limit):
    if value < 0:
        return value + limit
    elif value >= limit:
        return value - limit
    else:
        return value



def inner_product(Q, R):
    """
    Compute the MHD inner product in a uniformly
    sampled cylindrical geometry

    Parameters
    ----------

    Q: 2D numpy array of floats
    (6*n_samples = number of volume-sampled locations for each of the
    6 components of the two field vectors, M = number of time samples)
        Dimensionalized and normalized matrix of temporal BOD modes

    R: numpy array of floats
    (6*n_samples = number of volume-sampled locations for each of the
    6 components of the two field vectors, M = number of time samples)
        Radial coordinates of the volume-sampled locations

    Returns
    -------

    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The unscaled matrix of inner products X*X

    """
    Qr = np.zeros(np.shape(Q))
    for i in range(np.shape(Q)[1]):
        Qr[:, i] = Q[:, i] * np.sqrt(R)
    inner_prod = np.transpose(Qr) @ Qr
    return inner_prod
def vector_POD(inner_prod, t_train, r):
    from scipy.integrate import odeint
    from scipy.linalg import eig
    """
    Performs the vector POD, and scales the resulting modes
    to lie on the unit ball. Also returns the names of the
    temporal modes which will be modeled.

    Parameters
    ----------
    inner_prod: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The scaled matrix of inner products X*X

    t_train: 1D numpy array of floats
    (M_train = number of time samples in the training data)
        The time samples for training

    r: int
    (1)
        The truncation number of the SVD

    Returns
    -------
    x: 2D numpy array of floats
    (M = number of time samples, r = truncation number of the SVD)
        The temporal BOD modes to be modeled, scaled to
        stay on the unit ball

    feature_names: numpy array of strings
    (r = truncation number of the SVD)
        Names of the temporal BOD modes to be modeled

    S2: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The matrix of singular values

    Vh: 2D numpy array of floats
    (M = number of time samples, M = number of time samples)
        The V* in the SVD, returned here because the SINDy modes
        will need to be rescaled off of the unit ball to compare
        with the original measurements

    """

    # Compute eigenvalue decomposition of <Q,Q> and sort by largest eigenvalues
    S2, v = eig(inner_prod)
    idx = S2.argsort()[::-1]
    S2 = S2[idx]
    v = v[:, idx]
    Vh = np.transpose(v)

    # Plot temporal eigenvectors and associated eigenvalues
   
    print("% field in first r modes = ",
          sum(np.sqrt(S2[0:r])) / sum(np.sqrt(S2)))
    print("% energy in first r modes = ", sum(S2[0:r]) / sum(S2))

    # Reshape some things and normalize the temporal modes for SINDy
    S2 = np.diag(S2)
    vh = np.zeros((r, np.shape(Vh)[1]))
    feature_names = []
    for i in range(r):
        vh[i, :] = Vh[i, :] / sum(np.amax(abs(Vh), axis=1)[0:r])
        feature_names.append(r'$\varphi_{:d}$'.format(i + 1))
    x = np.transpose(vh)

    return x, feature_names, S2, Vh

