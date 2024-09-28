# Other decoders dont decode DSM outputs well, only dropoutNet - verified with walled environment

# from sklearn.linear_model import Ridge
# from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import matplotlib.pyplot as plt

# import ratinabox
# ratinabox.stylize_plots(); # ratinabox.autosave_plots=True; ratinabox.figure_directory="../figures/"

def train_decoder(Neurons, Ag=None, env_coords=None, method='dropoutNet'):
    ' different from ratinabox v - directly using the PC.get_state over discretized env instead of training using the neuron history '
    # Get training data

    if env_coords is None:
        if Ag is None:
            raise ValueError("Either 'Ag' or 'env_coords' must be provided.")
        env_coords = Ag.Environment.discretise_environment(dx=0.01) #(10000, 2)
        env_coords = env_coords.reshape(-1, env_coords.shape[-1])

    # inputs_all = PCs.get_state(evaluate_at="all").T # N of 10000 values corresponding to ag.Environment.flattened_discrete_coords # len 10000
    fr = Neurons.get_state(evaluate_at=None, pos=env_coords).T  # firing rate
    
    print('env_coords shape to train decoder: ',env_coords.shape)
    # Initialise and fit model
    if method=='LR' or method=='all':
        model_LR = Ridge(alpha=0.01)
        model_LR.fit(fr, env_coords)
        Neurons.decoding_model_LR = model_LR  # Save models into Neurons class for later use 
        print('linear_model trained')

    if method=='GP'or method=='all':
        from sklearn.gaussian_process.kernels import RBF
        model_GP = GaussianProcessRegressor(
            alpha=0.01,
            kernel=RBF(
                1
                * np.sqrt(
                    Neurons.n / 20
                ),  # <-- kernel size scales with typical input size ~sqrt(N)
                length_scale_bounds="fixed",
            ),
            n_restarts_optimizer=5
        )
        model_GP.fit(fr, env_coords)  
        
        Neurons.decoding_model_GP = model_GP  # Save models into Neurons class for later use 
        print('GaussianProcessRegressor trained')
    if method=='dropoutNet':
        train_xy_Net(Neurons, Ag=Ag, env_coords=env_coords)
        print('dropoutNet trained')
    return 

def decode_position(Neurons, PC_activities, plot=False,len_pointstoplot=None, method='dropoutNet',return_std=False, nolims=False):
    """  Returns a list of times and decoded positions"""
    assert PC_activities.ndim == 2, "PC_activities_trajectory should be 2D"
    if return_std:
        if (not hasattr(Neurons, 'decoding_model_GP') or Neurons.decoding_model_GP is None) and (not hasattr(Neurons, 'decoding_model_dropoutNet') or Neurons.decoding_model_dropoutNet is None):
            raise ValueError("Gaussian Process decoder and dropoutNet not trained. Train one decoder first.")

        # if Neurons.decoding_model_GP is None and Neurons.decoding_model_dropoutNet is None:
        #     raise ValueError("Gaussian Process decoder and dropoutNet not trained. Train one decoder first.")
        # raise ValueError("Gaussian Process decoder not trained. Train the decoder first.")
        if method=='all':
            raise ValueError("Either 'method' should be selected or 'get_uncertainty' must be False.")
    # positions_all= []
    # for PCs_activity in PC_activities:
    #     # decode position from the data and using the decoder saved in the  Neurons class
    #     decoded_pomethod=='GP'or method=='all'sition_GP = Neurons.decoding_model_GP.predict(PCs_activity.reshape(1,-1))
    #     # decoded_position_LR = Neurons.decoding_model_LR.predict(PCs_activity)
    #     positions_all.append(decoded_position_GP)
    # return positions_all #(decoded_position_GP, decoded_position_LR)

    # Decode positions from the data using the decoder saved in the Neurons class
    if method=='LR'or method=='all':
        decoded_position_LR = Neurons.decoding_model_LR.predict(PC_activities)
        positions_all = decoded_position_LR
    if method=='GP'or method=='all':
        if return_std:
            decoded_position_GP, std_prediction = Neurons.decoding_model_GP.predict(PC_activities, return_std=return_std)
        else:          
            decoded_position_GP = Neurons.decoding_model_GP.predict(PC_activities)
        positions_all = decoded_position_GP 

    if method=='dropoutNet'or method=='all':
        n_iter=10
        model = Neurons.decoding_model_dropoutNet 
        data_tensor = torch.from_numpy(np.array(PC_activities)).float()
        if return_std:
            model.train()  # Ensure dropout is active
            predictions_all = []
            # data_tensor = torch.tensor(PC_activities.tolist()) 
            for _ in range(n_iter):
                outputs = model(data_tensor).detach().numpy()
                predictions_all.append(outputs)
            predictions_all = np.array(predictions_all)
            decoded_position_dropoutNet = np.mean(predictions_all, axis=0)
            std_prediction = np.std(predictions_all, axis=0)
        else:
            model.eval()
            decoded_position_dropoutNet = model(data_tensor).detach().numpy()
        positions_all = decoded_position_dropoutNet

    # Convert the decoded_positions_GP array to a list of arrays
    # positions_all = [np.array([pos]) for pos in decoded_positions_GP]
    # positions_all = np.vstack(positions_all)

    if plot==True:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        if len_pointstoplot is None:
            len_pointstoplot = len(positions_all)
        xs = positions_all[:len_pointstoplot,0]
        ys = positions_all[:len_pointstoplot,1]
        colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1
        orig_map=plt.cm.get_cmap('viridis') 
        reversed_map = orig_map.reversed() 
        scatter = axs.scatter(xs, ys, alpha=0.3, s=10,c=colors, cmap=reversed_map)  # Use the numbers as colors
        axs.scatter(xs[0], ys[0], color='k',s=20)
        axs.scatter(xs[-1], ys[-1], color='red',s=25)
        axs.plot(xs, ys, color='blue',alpha=0.1)
        if not nolims:
            axs.set_xlim([0,1])
            axs.set_ylim([0,1])

    if method=='all':
        return (decoded_position_GP, decoded_position_LR, decoded_position_dropoutNet)
    if return_std:
        return positions_all, std_prediction
    return positions_all

 


# env_coords = Ag.Environment.discretise_environment(dx=0.1) # dx=Ag.environment.scale/10
# env_coords = env_coords.reshape(-1, env_coords.shape[-1])
# # inputs_all = PCs.get_state(evaluate_at="all").T # N of 10000 values corresponding to ag.Environment.flattened_discrete_coords # len 10000
# inputs_all = PCs.get_state(evaluate_at=None, pos=env_coords).T
# inputs_all.shape
# pc_all_env_coords=inputs_all

def simple_decode_position(PC_activities_trajectory, env_coords, pc_all_env_coords,plot=False):

    positions_all = np.empty((len(PC_activities_trajectory), env_coords.shape[1]))

    # Use NumPy's vectorized operations to find sel_val_indices and calculate the average
    for i, PCs_activity in enumerate(PC_activities_trajectory):
        is_close = np.isclose(pc_all_env_coords, PCs_activity, atol=1e-1)
        sel_val_indices = np.where(np.all(is_close, axis=1))[0]
        positions_all[i] = np.average(env_coords[sel_val_indices], axis=0)
 

    if np.isnan(positions_all).any():
        print("ARRAY HAS NAN VALS - INVALID, TRY DIFF DX FOR DISCRETIZING ENV")
        # positions_all = positions_all[~np.isnan(positions_all).any(axis=1)]
    if plot==True:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        len_pointstoplot = len(positions_all)
        xs = positions_all[:len_pointstoplot,0]
        ys = positions_all[:len_pointstoplot,1]
        colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1
        orig_map=plt.cm.get_cmap('viridis')
        reversed_map = orig_map.reversed() 
        scatter = axs.scatter(xs, ys, alpha=0.2, c=colors, cmap=reversed_map)  # Use the numbers as colors
        axs.scatter(xs[0], ys[0], color='k',s=20)
        axs.scatter(xs[-1], ys[-1], color='red',s=25)
        axs.plot(xs, ys, color='blue',alpha=0.2)
        axs.set_xlim([0,1])
        axs.set_ylim([0,1])

    return positions_all

    # positions_all= []
    # for PCs_activity in PC_activities_trajectory:
    #     sel_val_indices = [i for i, val in enumerate(pc_all_env_coords) if np.isclose(val, PCs_activity, atol=1e-1).all()]
    #     positions_all.append( np.average(env_coords[sel_val_indices],axis=0) )
    # return positions_all

# dataset_PCactivity1 = dataset.observation[0]
# %timeit simple_decode_position(dataset_PCactivity1.reshape(1, -1))  
# train_decoder(PCs)
# %timeit decode_position(PCs,dataset_PCactivity1.reshape(1, -1))        scatter = axs.scatter(xs, ys, alpha=0.2, c=colors, cmap='viridis')  # Use the numbers as colors




################################################################

# def get_uncertainties(Neurons, PC_activities): #,return_likelihood=False 
#     """  Returns uncertainties of the decoded positions - GP - SD or likelihood / prob"""
#     # if return_likelihood:
#     #     assert y_train is not None, "y_train must be provided to compute the likelihood"
#     assert PC_activities.ndim == 2, "PC_activities_trajectory should be 2D"
#     # Decode positions from the data using the decoder saved in the Neurons class
#     # for PCs_activity in PC_activities_trajectory:
#     #     decoded_position_GP = Neurons.decoding_model_GP.predict(PCs_activity.reshape(1,-1))
#     #     # decoded_position_LR = Neurons.decoding_model_LR.predict(PCs_activity)
#     #     positions_all.append(decoded_position_GP)
#     # return positions_all #(decoded_position_GP, decoded_position_LR)
#     if Neurons.decoding_model_GP is None:
#         raise ValueError("Gaussian Process decoder not trained. Train the decoder first.")
#     decoded_position_GP, std_prediction = Neurons.decoding_model_GP.predict(PC_activities, return_std=True)

#     # if return_likelihood:
#     #     # Compute the log likelihood of the data under the model
#     #     y_std = std_prediction 
#     #     y_mean = decoded_position_GP
#     #     y_likelihood = -0.5 *np.log(y_std**2)* np.sum((y_train - y_mean)**2 / 2* y_std**2 )
#     #     y_likelihood = -0.5 * np.sum(((y_train - y_mean)**2 / (2 * y_std**2)) + np.log(y_std**2))
#     #     return y_mean, y_std, y_likelihood

#     return decoded_position_GP, std_prediction



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class dropoutNet(nn.Module):
    def __init__(self):
        super(dropoutNet, self).__init__()
        self.fc1 = nn.Linear(50, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_xy_Net(Neurons, Ag=None, env_coords=None, epochs=30):
    # Get training data
    if env_coords is None:
        if Ag is None:
            raise ValueError("Either 'Ag' or 'env_coords' must be provided.")
        env_coords = Ag.Environment.discretise_environment(dx=0.01) #(10000, 2)
        env_coords = env_coords.reshape(-1, env_coords.shape[-1]).astype(np.float32)
    # inputs_all = PCs.get_state(evaluate_at="all").T # N of 10000 values corresponding to ag.Environment.flattened_discrete_coords # len 10000
    fr = Neurons.get_state(evaluate_at=None, pos=env_coords).T.astype(np.float32)  # firing rate

    model = dropoutNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    data_50d_tensor = torch.tensor(fr)
    data_2d_tensor = torch.tensor(env_coords)
    epochs = epochs
    batch_size = 32

    for epoch in range(epochs):
        permutation = torch.randperm(data_50d_tensor.size()[0])
        for i in range(0, data_50d_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_50d_tensor[indices], data_2d_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x).float()
            
            try:
                loss = criterion(outputs, batch_y.float())
                loss.backward()
            except RuntimeError as e:
                if str(e) == "Found dtype Double but expected Float":
                    print('debug: ',str(e))
                    break
            optimizer.step()
        # print(loss.item())
            
    Neurons.decoding_model_dropoutNet = model


# # Predict with uncertainty
# predictions = predict_with_uncertainty(model, test_data)
# mean_predictions = np.mean(predictions, axis=0)
# std_predictions = np.std(predictions, axis=0)
# # Predictive entropy
# predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=1)
# # Mutual Information (Epistemic Uncertainty)
# expected_entropy = -np.mean(np.sum(predictions * np.log(predictions + 1e-10), axis=2), axis=0)
# mutual_information = predictive_entropy - expected_entropy