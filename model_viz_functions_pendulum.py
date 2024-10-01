# test changes
# import importlib
# importlib.reload(modelviz_pendulum_utils)

from dsm.state import FittedValueTrainState

import jax
import jax.numpy as jnp

import einops
import numpy as np

from tqdm import tqdm
####################################################################


def compute_activation_layer_1source(model_generator, atom_params, source_state,zs, layers):
    """ For a single source state, computes the activation of the specified layer (single) of the atom
    and returns the activations. input to the the atom is the latent vector zs (of dim num_latent) concatenated 
    with the source state (of dim num_state_dims)."""
    # zs = jax.random.normal(rng, (num_latent_dims,))
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
    activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
    # print('activations: ',activations_layer)
    return activations_layer

def compute_activation_layer_all_sources(model_generator, atom_params, sources_all,zs, layers):
    """ For all source states, computes the activation of the specified layer (single) of the atom
    and returns the activations. input to the the atom is the latent vector zs (of dim num_latent) concatenated 
    with the source state (of dim num_state_dims)"""
    activations_all = []
    for source_state in sources_all:
        activations_all.append(compute_activation_layer_1source(model_generator, atom_params, 
                                                                source_state, zs, layers))
    return np.array(activations_all)

####################################################################

def compute_activation_allintermediates_1source(model_generator, atom_params, source_state, zs, layers):
    """ For a single source state, computes the activation of the specified layers (multiple) of the atom, 
    input to the the atom is the latent vector zs (of dim num_latent) concatenated 
    with the source state (of dim num_state_dims)"""
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    _, intermediates = predict_with_all_intermediates(model_generator, atom_params, xs_atom)
    
    activations_layers = {}
    for layer in layers:
        activations_layers[layer] = intermediates['intermediates'][layer]['__call__'][0]
    return activations_layers


def compute_activation_allintermediates_all_sources(model_generator, atom_params, sources_all, zs, layers):
    """ For all source states, computes the activation of the specified layers (multiple) of the atom,"""
    activations_all = {layer: [] for layer in layers}
    
    for source_state in sources_all:
        activations_source = compute_activation_allintermediates_1source(model_generator, atom_params, 
                                                                         source_state, zs, layers)
        for layer in layers:
            activations_all[layer].append(activations_source[layer])
    # Convert lists to numpy arrays
    for layer in layers:
        activations_all[layer] = np.array(activations_all[layer])
    
    return activations_all
####################################################################

def predict_with_all_intermediates(model_generator,atom_params, x): 
    """ For input x (dim=num_state_dims+num_latent), predicts the output 
    of the selected atom (whose params are passed) and returns all the intermediate outputs."""
    # mlp_model = MLP(num_layers=3, num_hidden_units=8, num_outputs=4)
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = model_generator.apply_fn.__self__.num_state_dims #3
    outputs, intermediates = mlp_model_atom.apply({"params": atom_params}, x, capture_intermediates=True)
    return outputs, intermediates

def predict_layer(model_generator, atom_params, x, layers):
    """ For input x (dim=num_state_dims+num_latent), predicts the output of the selected atom (whose params are passed) 
    and return the intermediate outputs for the specified layers."""
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = model_generator.apply_fn.__self__.num_state_dims #3
    # filter specifically for only the layers like Dense_0 and Dense_2 
    filter_fn = lambda mdl, method_name: isinstance(mdl.name, str) and (mdl.name in layers )
    return mlp_model_atom.apply({"params": atom_params}, x, capture_intermediates=filter_fn)


####################################################################

# plotting generated samples
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

####################################################################

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import gridspec

cmap_ratemap = 'inferno'         
def plot_neuron_activations(activations_layer, neuron_idx, figlabel, xpos,ypos, ax_main=None, normalize=False,debug=False):
    """ Contour plot of the activations of a single neuron or multiple neurons in a layer."""
    # xpos = angles, ypos = z
    if isinstance(neuron_idx, str) and neuron_idx == 'all':
        neuron_idx = np.arange(activations_layer.shape[-1])
    
    # plot for single neuron
    if isinstance(neuron_idx, int): #if not isinstance(neuron_idx,np.ndarray): 
        # print('debug activations_layer', activations_layer.shape)
        activations_selected = activations_layer[:, neuron_idx]
        if debug:
            print('debug activations_selected', activations_selected.min(), activations_selected.max())
        # outputs_reshaped = activations_selected.reshape(X.shape)
        if normalize:
            activations_selected = (activations_selected - np.amin(activations_selected)) / (np.amax(activations_selected) - np.amin(activations_selected))

        if ax_main==None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
        else:
            ax = ax_main
        activations_2d = activations_selected.reshape(len(ypos), len(xpos))
        scatter = ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
        ax.set_aspect('equal')
        # scatter = ax.scatter(angles_flat, Z_flat, c=activations_selected) #, cmap='viridis'
        ax.set_title(f'{figlabel}')
        ax.figure.colorbar(scatter, ax=ax, label='Neuron Activity')
        if ax_main==None:
            fig.colorbar(scatter, ax=ax, label='Neuron Activity')
            # Set labels
            ax.set_xlabel('θ')
            ax.set_ylabel('θdot')
            plt.show()

    elif isinstance(neuron_idx,np.ndarray):
        # plot for multiple neurons in same layer
        # fig = plt.figure()
        if isinstance(neuron_idx,np.ndarray):
            Tot = len(neuron_idx)
            if Tot % 10 == 0:
                Cols = 10
            else:
                Cols = 8 
            if Tot < Cols:
                Cols = Tot
        else:
            Tot = len(activations_layer)
            Cols = 5
        Rows = Tot // Cols 
        if Tot % Cols != 0:
            Rows += 1
        # Position = range(1,Tot + 1)
        # print(activations_layer.shape, Tot)
        if isinstance(neuron_idx,np.ndarray):
            gs = gridspec.GridSpec(Rows, Cols+1, width_ratios=[*([1]*Cols), 0.1])
            if Tot>10:
                fig = plt.figure(figsize=(11, 10))
            else:
                fig = plt.figure()
            if normalize:
                vmin, vmax = activations_layer.min(), activations_layer.max()
            scatters = []
            # fig, axes = plt.subplots(nrows=Rows,ncols=Cols, constrained_layout=True)  #new
            # if Rows == 1:
            #     axes = axes[np.newaxis, :]
            # elif Cols == 1:
            #     axes = axes[:, np.newaxis]
            
            for k,neuron_i in enumerate(neuron_idx):
                activations_selected = activations_layer[:, neuron_i]
                # print(activations_selected.shape)            
                # ax = fig.add_subplot(Rows,Cols,Position[k])
                # ax = axes[k//Cols, k % Cols] #new
                ax = fig.add_subplot(gs[k//Cols, k % Cols])

                activations_2d = activations_selected.reshape(len(ypos), len(xpos))
                if normalize:
                    contour_set = ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20, vmin=vmin, vmax=vmax)
                else:
                    contour_set = ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
                scatters.append(contour_set)
                ax.set_box_aspect(1)  #new
                # ax.set_aspect('equal')
                ax.set_xlabel('θ', labelpad=0)
                ax.set_ylabel('θdot', labelpad=0)
                if Tot<=5:
                    # # cbar = plt.colorbar(contour_set, ax=ax)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="5%", pad=0.1)  # Adjusted the colorbar size to 5%
                    cbar = fig.colorbar(contour_set, cax=cax, orientation="horizontal", ticks = [activations_2d.min(), activations_2d.max()]) #cbar = 
                    cbar.ax.xaxis.set_ticks_position('top')
                    plt.subplots_adjust(wspace=0.8, hspace=0.2)
                if Tot>10:
                    # ax.set_yticks([])
                    # ax.set_xticks([])
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                    ax.yaxis.set_major_locator(ticker.NullLocator())
            # scatter = ax.scatter(angles_flat, Z_flat, c=activations_selected)

            # ax = fig.add_subplot(Rows,Cols,Position[k], projection='3d')
            # scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=activations_selected)
        
            if Tot>10 and normalize:
                # fig.subplots_adjust(right=0.8)
                # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar_ax = fig.add_subplot(gs[:, -1])
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_ratemap, norm=norm, label='Neuron Activity')
                # cbar = plt.colorbar(scatters, ax=ax)
                # fig.colorbar(scatters[0], ax=fig.get_axes(), label='Theta Activity')

        else:
            raise ValueError('neuron_idx should be an integer or numpy array or neuron_idx == avg')
        # plt.tight_layout() #new
        fig.suptitle(f'{figlabel}')
####################################################################

def plot_neuron_ratemaps_from_activationlayerlist(activations_layer_list, figlabel, neuron_idx, xpos, ypos, normalize=False,debug=False,title=None):
    n = len(activations_layer_list)
    cols = 5 
    rows = n // cols 
    rows += n % cols
    position = range(1,n + 1)
    # fig = plt.figure(figsize=(8, 8))
    fig, axes = plt.subplots(rows,cols, figsize=(15, 6), constrained_layout=True)
    if rows == 1:
        axes = axes[np.newaxis, :]
    for k, activations_layer in enumerate(activations_layer_list): 
        ax = axes[k//cols, k % cols]
        # ax = fig.add_subplot(rows, cols, position[k])
        # ax.set_aspect('equal')
        label = f'{figlabel}: {k}'
        plot_neuron_activations(activations_layer,neuron_idx,label,xpos,ypos, ax_main=ax, normalize=normalize,debug=debug)
        plt.xticks([]) 
        plt.yticks([]) 
        ax.set_xlabel('θ')
        ax.set_ylabel('θdot')
        ax.set_box_aspect(1)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    if title is not None:
        fig.suptitle(title)
    plt.show()

####################################################################


####################################################################
# plot generated samples directly (for pendulum env)
# img_dict = plotting.plot_samples(state.generator, jax.random.PRNGKey(0),config=config)
# src_idx = 0
# plot_key = list(img_dict.keys())[src_idx]
# sample_plotarray = img_dict[plot_key]
# plt.imshow(sample_plotarray)

from dsm import datasets
def plot_samples_dsm(samples,source, make_dataset=False,atom='all'):
    """ Visualize the generated samples in the Pendulum environment. Also the original dataset the model was trained on."""
    # for Pendulum env
    if make_dataset:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3 * 2, 3))
        # # Left scatter plot
        # # converts Cartesian coordinates to polar coordinates (thetas) and extracts velocities
        ENVIRONMENT = "Pendulum-v1"
        dataset = datasets.make_dataset(ENVIRONMENT)
        thetas = np.arctan2(dataset.observation[:, 1], dataset.observation[:, 0]) % (2 * np.pi)
        velocities = dataset.observation[:, -1]

        # colors = range(len(thetas))
        axs[0].scatter(thetas, velocities, alpha=0.1, s=1, color="grey")
        # axs[0].scatter(xs[0], ys[0], color='k',s=20)
        # axs[0].scatter(thetas[-1], velocities[-1], color='red',s=20)
        # axs.plot(xs, ys, color='blue',alpha=0.2)
        # axs.set_xlim([0,1])
        # axs.set_ylim([0,1])
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    # Plot atom scatter & kde
    # plots generated samples - each colour represents atom?

    # cmap = plt.get_cmap("Dark2") 
    cmap = plt.get_cmap("tab10") 
    if isinstance(atom, int):
        thetas = np.arctan2(samples[atom, :, 1], samples[atom, :, 0]) % (2 * np.pi)
        velocities = samples[atom, :, -1]

        if make_dataset:
            axs[1].scatter(thetas, velocities, color=cmap(atom), s=30, alpha=0.25)
        else:
            plt.scatter(thetas, velocities, color=cmap(atom), s=30, alpha=0.25)
        plt.title(f"Generated samples of atom {atom}")

    elif atom=='all': #elif isinstance(atom, np.ndarray):
        atom_idx_values = np.arange(samples.shape[0])
        norm = Normalize(vmin=min(atom_idx_values), vmax=max(atom_idx_values))
        sm = ScalarMappable(cmap=cmap, norm=norm)

        for atom_idx in atom_idx_values:
            thetas = np.arctan2(samples[atom_idx, :, 1], samples[atom_idx, :, 0]) % (2 * np.pi)
            velocities = samples[atom_idx, :, -1]
            if make_dataset:
                # axs[1].scatter(xs, ys,c=colors,s=40, cmap=cmap, alpha=0.25)
                axs[1].scatter(thetas, velocities, color=cmap(norm(atom_idx)),s=40, alpha=0.25)
            else:
                # plt.scatter(xs, ys,c=colors,s=40, cmap=cmap, alpha=0.25)
                plt.scatter(thetas, velocities, color=cmap(norm(atom_idx)),s=40, alpha=0.25)
        ax = axs[1] if make_dataset else plt.gca()
        plt.colorbar(sm, ax=plt.gca(), label='atom_idx')
        plt.title("Generated samples of all atoms")
    
    # Plot source state
    sourcetheta = np.arctan2(source[1], source[0]) % (2 * np.pi)
    # print('debug source theta:',sourcetheta)
    if make_dataset:
        for ax in axs:
            ax.scatter(sourcetheta, source[-1], marker="x", s=32, alpha=0.8, color="red")
    else:
        plt.scatter(sourcetheta, source[-1], marker="x", s=64, alpha=0.8, color="red")

    # set bounds
    if make_dataset:
        for ax in axs:
            ax.set_ylim(-8.5, 8.5)
            ax.set_aspect("auto")
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    else:
        plt.ylim(-8.1, 8.1) 
        plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    # image = plotting_utils.fig_to_ndarray(fig)
    plt.show(fig)
####################################################################
import model_viz_loaders
def plot_predictions_atoms(config, latent_rng_seed, model_generator, source_states, 
                             atoms_sel = 'all', title=None):
    """ Given the source states, and selected atoms, gets the model's predictions (layer 3 activation) and plots them."""
    layers = ['Dense_3',]  # predicted_pcs same as 3rd layer activations
    zs = jax.random.normal(jax.random.PRNGKey(latent_rng_seed), (config.latent_dims,)) 
    if atoms_sel == 'all':
        atoms_sel = range(config.num_outer)
    else:
        assert isinstance(atoms_sel, list)
    preds_all_atoms_list = []
    for atom_idx in tqdm(atoms_sel, desc='Processing atoms'):
        # print(atom_idx)
        atom_params = model_viz_loaders.extract_params_ith_atom(model_generator, atom_idx, config.num_outer)
        # model_viz_loaders.compute_activation_layer_all_sources
        preds_all_atom = compute_activation_layer_all_sources(model_generator, atom_params, source_states,zs, layers)
        assert preds_all_atom.shape == source_states.shape
        thetas = np.arctan2(preds_all_atom[:, 1], preds_all_atom[:, 0]) % (2 * np.pi)
        velocities = preds_all_atom[:, -1]
        preds_all_atoms_list.append(np.array([thetas, velocities]).T)

    # return preds_all_atoms_list

    n = len(preds_all_atoms_list)
    cols = 5 
    rows = n // cols 
    rows += n % cols
    position = range(1,n + 1)
    # fig = plt.figure(figsize=(15, 6)) #figsize=(10, 10)
    fig, axes = plt.subplots(rows,cols, figsize=(15, 6), constrained_layout=True)
    for k, positions_pred in enumerate(preds_all_atoms_list): 
        # ax = fig.add_subplot(rows, cols, position[k])
        ax = axes[k//cols, k % cols]
        # ax.set_aspect('equal')
        xs, ys = positions_pred[:,0], positions_pred[:,1]
        colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1

        ax.scatter(xs, ys, alpha=0.2, c=colors, cmap='viridis')  # Use the numbers as colors
        ax.scatter(xs[0], ys[0], color='k',s=20)
        ax.scatter(xs[-1], ys[-1], color='red',s=25)
        # ax.plot(xs, ys, color='blue',alpha=0.2)
        ax.set_xlabel('θ')
        ax.set_ylabel('θdot')

        ax.set_xlim([0,2*np.pi])
        ax.set_ylim([-8.5,8.5])
        ax.set_box_aspect(1)
        # ax.set_aspect("auto")
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
        # plt.xticks([]) 
        # plt.yticks([]) 
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    if title is not None:
        fig.suptitle(title)
    plt.show() 


#############


################################################################################################################


################################################################################################################ 
# xs_atom = xs[0][2]
# layers = {'Dense_3',} #, 'Dense_2'
# preds, feats = predict_layer(atom_params, xs_atom,layers)
# print(preds)
# print(feats)
# preds, intermediates = predict_with_all_intermediates(atom_params, xs_atom)
# # print(intermediates['intermediates'].keys())  # This will contain the intermediate outputs
# # feats # intermediate c in Model is stored and isn't filtered out by the filter function
################################################################################################################




# def multiple_samples_compute_activation_layer_1source(model_generator, atom_params, source_state, layers, 
#                                                       num_samples, num_latent_dims, random_seed = 0 ):
#     context = einops.repeat(source_state, "s -> i s", i=num_samples)
#     zs = jax.random.normal(jax.random.PRNGKey(random_seed), (num_samples, num_latent_dims))
#     xs_all = jnp.concatenate((zs, context), axis=-1)

#     all_samples_activations= []
#     for xs_atom in xs_all:
#         preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
#         activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
#         all_samples_activations.append(activations_layer)
#     all_samples_activations = np.array(all_samples_activations)

#     return all_samples_activations

# def show_allsamples_compute_activation_layer_all_sources(model_generator, atom_params, sources_all, layers,num_samples,num_latent_dims):
#     activations_all = []
#     for source_state in sources_all:
#         activations_all.append(multiple_samples_compute_activation_layer_1source(model_generator, atom_params, 
#                                                                                  source_state, layers,num_samples,num_latent_dims))
#     return np.array(activations_all)



def predict_output(model_generator, atom_params, x):
    """ For input x (dim=num_state_dims+num_latent), predicts the output of the selected atom (whose params are passed) 
    and return the intermediate outputs for the specified layers."""
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = model_generator.apply_fn.__self__.num_state_dims #3
    # filter specifically for only the layers like Dense_0 and Dense_2 
    # filter_fn = lambda mdl, method_name: isinstance(mdl.name, str) and (mdl.name in layers )
    # return mlp_model_atom.apply({"params": atom_params}, x, capture_intermediates=filter_fn)
    return mlp_model_atom.apply({"params": atom_params}, x)

def get_multiple_predictions_atoms(config, latent_rng_seed, model_generator, source_states, 
                           atoms_sel = 'all', num_samples=2, title=None):
    """ Given the source states, and selected atoms, gets the model's predictions (layer 3 activation) - multiple of them """
    zs = jax.random.normal(jax.random.PRNGKey(latent_rng_seed), (num_samples, config.latent_dims,))
    if atoms_sel == 'all':
        atoms_sel = range(config.num_outer)
    else:
        assert isinstance(atoms_sel, list)
    preds_all_atoms_list = []
    debug_shape = 1
    for atom_idx in tqdm(atoms_sel, desc='Processing atoms'):
        atom_params = model_viz_loaders.extract_params_ith_atom(model_generator, atom_idx, config.num_outer)
        activations_all = []
        for source_state in source_states:
            preds_for_state = []
            context = einops.repeat(source_state, "s -> i s", i=num_samples)
            xs_all = jnp.concatenate((zs, context), axis=-1)

            all_samples_activations= []
            for xs_atom in xs_all:
                # preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
                # activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
                # all_samples_activations.append(activations_layer)
                preds = predict_output(model_generator, atom_params, xs_atom)
                # print(preds)
                preds_for_state.append(preds)
            all_samples_activations = np.array(all_samples_activations)                
            activations_all.append(preds_for_state)
        
        preds_all_atom = np.array(activations_all)
        if debug_shape:
            print('debug shape: ',preds_all_atom.shape)
            debug_shape=0
        
        # Extract thetas and velocities
        thetas = np.arctan2(preds_all_atom[:, :, 1], preds_all_atom[:, :, 0]) % (2 * np.pi)
        velocities = preds_all_atom[:, :, -1]
        preds_all_atoms_list.append(np.array([thetas, velocities]).T)
    
    return preds_all_atoms_list


def plot_multiple_predictions_atoms(preds_all_atoms_list, num_samples=10, title=None):
    """ Plots the predictions from all atoms, showing 5 samples per source state."""
    n = len(preds_all_atoms_list)
    cols = 5  # Number of columns per source state (for each sample)
    rows = n  // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6), constrained_layout=True)
    if rows == 1:
        axes = axes[np.newaxis, :]
    for k, positions_pred in enumerate(preds_all_atoms_list): 
        ax = axes[k//cols, k % cols]
        for i in range(num_samples):
            xs, ys = positions_pred[i][:, 0], positions_pred[i][:, 1]
            colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1
            
            ax.scatter(xs, ys, alpha=0.2, c=colors, cmap='viridis')  # Use the numbers as colors
            ax.scatter(xs[0], ys[0], color='k', s=20)
            ax.scatter(xs[-1], ys[-1], color='red', s=25)
            ax.set_xlim([0, 2*np.pi])
            ax.set_ylim([-8.5, 8.5])
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
            ax.set_xlabel('θ')
            ax.set_ylabel('θdot')
            ax.set_box_aspect(1)
            ax.set_title(f'Atom{k}')
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    plt.show()


####################
# # normalized
# Tot=3
# Cols=3
# Rows=1
# normalize=True
# xpos, ypos=theta_all,thetadot_all
# from matplotlib import gridspec
# gs = gridspec.GridSpec(Rows, Cols+1, width_ratios=[*([1]*Cols), 0.1])
# fig = plt.figure()
# activations_layer3 = atom_activations_alllayers['Dense_3']
# if normalize:
#     # vmin, vmax = activations_layer3.min(), activations_layer3.max()
# for k,neuron_i in enumerate(activations_layer3.T):
#     ax = fig.add_subplot(gs[k//Cols, k % Cols])
#     activations_2d = neuron_i.reshape(len(ypos), len(xpos)) 
#     if k==1 or k==2:
#         vmin, vmax = activations_2d.min(), activations_2d.max()
#     contour_set = ax.contourf(xpos, ypos, activations_2d, cmap="inferno", levels=20, vmin=vmin, vmax=vmax)
#     scatters.append(contour_set)
#     ax.set_box_aspect(1)  #new
#     ax.set_xlabel('θ', labelpad=0)
#     ax.set_ylabel('θdot', labelpad=0)
#     # # cbar = plt.colorbar(contour_set, ax=ax)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=0.1)  # Adjusted the colorbar size to 5%
#     cbar = fig.colorbar(contour_set, cax=cax, orientation="horizontal", ticks = [activations_2d.min(), activations_2d.max()]) #cbar = 
#     cbar.ax.xaxis.set_ticks_position('top')
#     plt.subplots_adjust(wspace=0.8, hspace=0.2)
# plt.show()


#####################################################################

# # Exploring actions and rewards of the 10 atoms for the samples above from  one source state

# # plotting utils
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
# atom_idx_values = np.arange(samples.shape[0])
# norm = Normalize(vmin=min(atom_idx_values), vmax=max(atom_idx_values))
# sm = ScalarMappable(cmap='tab10', norm=norm)

# policy = datasets.make_policy(config.env)
# def apply_policy_to_batch(batch):
#     keys,samples = batch
#     return jax.lax.map(lambda x: policy(x[0],x[1]),(keys,samples))
# keys = jax.random.split(jax.random.PRNGKey(0), np.prod(samples.shape[:-1]))
# keys = jnp.array(keys).reshape(*samples.shape[:-1], -1)
# actions = jax.lax.map(apply_policy_to_batch, (keys,samples))[1]
# print('debug actions min max',np.min(actions), np.max(actions))

# # # Assuming actions[1] has shape (10, 64, 1, 1)
# # # Reshape it to (10, 64)
# # reshaped_actions = actions.reshape(actions.shape[0], -1)
# # # for i in range(10):
# # #     plt.scatter([i]*len(reshaped_actions[i]), reshaped_actions[i])
# # x_coords = np.repeat(np.arange(reshaped_actions.shape[0]), reshaped_actions.shape[1])
# # # Flatten reshaped_actions) for plotting
# # flat_actions = reshaped_actions.flatten()
# # plt.scatter(x_coords, flat_actions, c=x_coords, alpha=0.5, cmap='viridis')
# # plt.xlabel('Atom index')
# # plt.ylabel('Action value')
# # plt.show()

# rewards = jax.vmap(jax.vmap(reward_fn))(samples, actions).squeeze()

# reshaped_rewards = rewards.reshape(rewards.shape[0], -1)
# # for i in range(10):
# #     plt.scatter([i]*len(reshaped_actions[i]), reshaped_actions[i])
# x_coords = np.repeat(np.arange(reshaped_rewards.shape[0]), reshaped_rewards.shape[1])
# # Flatten reshaped_actions) for plotting
# flat_rewards = reshaped_rewards.flatten()
# plt.scatter(x_coords, flat_rewards, c=x_coords, alpha=0.5, cmap='viridis')
# plt.xlabel('Atom index')
# plt.ylabel('Reward value')
# plt.show()

# reward_mean = rewards.mean(axis=-1) / (1.0 - config.gamma)  
# reward_mean.shape
# import seaborn as sns
# sns.histplot(reward_mean, kde=True, color='blue')


###########################################################################

# # Plot with respect to theta arctan(neuron1 output, neuron2 output)
# # from mpl_toolkits.axes_grid1 import make_axes_locatable
# n = len(activations_layer_all_atoms)
# cols = 5 
# rows = n // cols 
# position = range(1,n + 1)
# fig = plt.figure(figsize=(10,4))

# # Find global min and max
# global_min = -np.pi
# global_max = np.pi
# scatters = []
# plt.subplots_adjust(wspace=0.5, hspace=0.5)

# for k, activations_layer in enumerate(activations_layer_all_atoms): 
#     ax = fig.add_subplot(rows, cols, position[k])
#     # ax.set_aspect('auto') #ax.set_box_aspect(1)  #ax.set_aspect('equal')
#     label = f'Atom: {k}'
#     theta = np.arctan2(activations_layer[:, 0], activations_layer[:, 1])
#     activations_2d = theta.reshape(len(thetadot_all), len(theta_all))
#     # scatter = ax.imshow(activations_2d, cmap='inferno', vmin=global_min, vmax=global_max, aspect='auto')
#     # ax.set_aspect('equal')
#     scatter = ax.contourf(theta_all, thetadot_all, activations_2d, cmap='inferno', levels=20, 
#                           vmin=global_min, vmax=global_max)
#     ax.set_box_aspect(1)
#     scatters.append(scatter)
#     ax.set_title(f'Atom{k}')
#     plt.xticks([]), plt.yticks([]) 
#     ax.set_xlabel('θ')
#     ax.set_ylabel('θdot')

# # Create colorbar as a common for all subplots.
# fig.colorbar(scatters[0], ax=fig.get_axes(), label='Predicted Orientation (θ)')
# # fig.suptitle('Theta predictions')
# plt.show()

#################################################################################

# # Plot with respect to angular velocity
# # single sample prediction from each atom in a subplot 

# n = len(activations_layer_all_atoms) # 10
# cols = 5 
# rows = n // cols 
# position = range(1,n + 1)
# fig = plt.figure(figsize=(10,4))
# # Find global min and max - angular velocity
# global_min = -8
# global_max = 8
# norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
# scatters = []
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# for k, activations_layer in enumerate(activations_layer_all_atoms): 
#     ax = fig.add_subplot(rows, cols, position[k])
#     label = f'Atom: {k}'
#     thetadot = activations_layer[:, 2]
#     activations_2d = thetadot.reshape(len(thetadot_all), len(theta_all))
#     # scatter = ax.imshow(activations_2d, cmap='inferno', vmin=global_min, vmax=global_max, aspect='auto')
#     # ax.set_aspect('equal')
#     scatter = ax.contourf(theta_all, thetadot_all, activations_2d, cmap='inferno', levels=20, norm=norm)
#     ax.set_box_aspect(1)
#     scatters.append(scatter)
#     ax.set_title(f'Atom{k}')
#     plt.xticks([]), plt.yticks([]) 
#     ax.set_xlabel('θ')
#     ax.set_ylabel('θdot')
# # Create colorbar as a common for all subplots.
# # fig.colorbar(scatters[0], ax=fig.get_axes(), label='Predicted Angular Velocity (θdot)', norm=norm)
# # Create a ScalarMappable for the colorbar with the global norm and colormap
# sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
# sm.set_array([])  # This is required to create the colorbar
# fig.colorbar(sm, ax=fig.get_axes(), label='Predicted Angular Velocity (θdot)')
# # fig.suptitle('Thetadot predictions')
# plt.show()

# # individual colorbars
# # neuron_idx=2 # angular velocity
# # modelviz_pendulum_utils.plot_neuron_ratemaps_from_activationlayerlist(activations_layer_all_atoms, 
# #                                                                       'Atom', neuron_idx, 
# #                                                                       theta_all, thetadot_all, normalize=False) 
# #                                                                         #,title='Velocity predictions'
