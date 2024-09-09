from dsm.state import FittedValueTrainState

import jax
import jax.numpy as jnp

import einops
import numpy as np

from tqdm import tqdm
#######################################
# plotting generated samples
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize




#######################################
# plot generated samples directly (for pendulum env)
from dsm import datasets
def plot_samples_dsm(samples,source, make_dataset=False,atom='all'):
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

    cmap = plt.get_cmap("Dark2") 
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
        plt.title("Decoded sample positions of all atoms")
    
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

################################################################################################################

def predict_with_all_intermediates(model_generator,params, x): 
    # mlp_model = MLP(num_layers=3, num_hidden_units=8, num_outputs=4)
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = model_generator.apply_fn.__self__.num_state_dims #3
    outputs, intermediates = mlp_model_atom.apply({"params": params}, x, capture_intermediates=True)
    return outputs, intermediates

def predict_layer(model_generator, params, x, layers):  #self, 
    mlp_model_atom = model_generator.apply_fn.__self__.model
    mlp_model_atom.num_outputs = model_generator.apply_fn.__self__.num_state_dims #3
    # filter specifically for only the layers like Dense_0 and Dense_2 
    filter_fn = lambda mdl, method_name: isinstance(mdl.name, str) and (mdl.name in layers )
    return mlp_model_atom.apply({"params": params}, x, capture_intermediates=filter_fn)

# xs_atom = xs[0][2]
# layers = {'Dense_3',} #, 'Dense_2'
# preds, feats = predict_layer(atom_params, xs_atom,layers)
# print(preds)
# print(feats)
# preds, intermediates = predict_with_all_intermediates(atom_params, xs_atom)
# # print(intermediates['intermediates'].keys())  # This will contain the intermediate outputs
# # feats # intermediate c in Model is stored and isn't filtered out by the filter function
################################################################################################################

def compute_activation_layer_1source(model_generator, atom_params, source_state,zs, layers):
    # zs = jax.random.normal(rng, (num_latent_dims,))
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    preds, feats = predict_layer(model_generator, atom_params, xs_atom,layers)
    activations_layer= feats['intermediates'][layers[0]]['__call__'][0]
    # print('activations: ',activations_layer)
    return activations_layer

def compute_activation_layer_all_sources(model_generator, atom_params, sources_all,zs, layers):

    activations_all = []
    for source_state in sources_all:
        activations_all.append(compute_activation_layer_1source(model_generator, atom_params, 
                                                                source_state, zs, layers))
    return np.array(activations_all)

#####################################

def compute_activation_allintermediates_1source(model_generator, atom_params, source_state, zs, layers):
    xs_atom = jnp.concatenate((zs, source_state), axis=-1)
    _, intermediates = predict_with_all_intermediates(model_generator, atom_params, xs_atom)
    
    activations_layers = {}
    for layer in layers:
        activations_layers[layer] = intermediates['intermediates'][layer]['__call__'][0]
    return activations_layers


def compute_activation_allintermediates_all_sources(model_generator, atom_params, sources_all, zs, layers):
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
################################################

import matplotlib.ticker as ticker
cmap_ratemap = 'inferno'         
def plot_neuron_activations(activations_layer, neuron_idx, figlabel, xpos,ypos, ax_main=None, normalize=False,debug=False):
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
            ax.set_xlabel('Angle')
            ax.set_ylabel('velocity')
            plt.show()

    elif isinstance(neuron_idx,np.ndarray):
        # plot for multiple neurons in same layer
        fig = plt.figure()
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
        Position = range(1,Tot + 1)
        # print(activations_layer.shape, Tot)
        if isinstance(neuron_idx,np.ndarray):
            for k,neuron_i in enumerate(neuron_idx):
                activations_selected = activations_layer[:, neuron_i]
                # print(activations_selected.shape)            
                ax = fig.add_subplot(Rows,Cols,Position[k])
                activations_2d = activations_selected.reshape(len(ypos), len(xpos))
                contour_set = ax.contourf(xpos, ypos, activations_2d, cmap=cmap_ratemap, levels=20)
                ax.set_aspect('equal')
                cbar = plt.colorbar(contour_set, ax=ax)
            # scatter = ax.scatter(angles_flat, Z_flat, c=activations_selected)

            # ax = fig.add_subplot(Rows,Cols,Position[k], projection='3d')
            # scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=activations_selected)
            # # plt.yticks([]) 
            # # plt.xticks([]) 
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.set_xlabel('Angle')
            ax.set_ylabel('velocity')
        else:
            raise ValueError('neuron_idx should be an integer or numpy array or neuron_idx == avg')
        fig.suptitle(f'{figlabel}')


###############################################
import model_viz_loaders

def plot_predictions_atoms(config, latent_rng_seed, model_generator, source_states, 
                             atoms_sel = 'all', title=None):
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
    fig = plt.figure() #figsize=(10, 10)
    for k, positions_pred in enumerate(preds_all_atoms_list): 
        ax = fig.add_subplot(rows, cols, position[k])
        ax.set_aspect('equal')
        xs, ys = positions_pred[:,0], positions_pred[:,1]
        colors = range(len(xs))  # Create a list of numbers from 0 to len(xs)-1
        
        ax.scatter(xs, ys, alpha=0.2, c=colors, cmap='viridis')  # Use the numbers as colors
        ax.scatter(xs[0], ys[0], color='k',s=20)
        ax.scatter(xs[-1], ys[-1], color='red',s=25)
        # ax.plot(xs, ys, color='blue',alpha=0.2)
        ax.set_xlim([0,2*np.pi])
        ax.set_ylim([-8.5,8.5])
        ax.set_aspect("auto")
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
        # plt.xticks([]) 
        # plt.yticks([]) 
    if title is not None:
        fig.suptitle(title)
    plt.show() 
            
##########################################

def plot_neuron_ratemaps_from_activationlayerlist(activations_layer_list, figlabel, neuron_idx, xpos, ypos, normalize=False,debug=False):
    n = len(activations_layer_list)
    cols = 5 
    rows = n // cols 
    rows += n % cols
    position = range(1,n + 1)
    fig = plt.figure(figsize=(8, 8))
    for k, activations_layer in enumerate(activations_layer_list): 
        ax = fig.add_subplot(rows, cols, position[k])
        ax.set_aspect('equal')
        label = f'{figlabel}: {k}'
        plot_neuron_activations(activations_layer,neuron_idx,label,xpos,ypos, ax_main=ax, normalize=normalize,debug=debug)
        plt.xticks([]) 
        plt.yticks([]) 
        ax.set_xlabel('Angle')
        ax.set_ylabel('velocity')
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    plt.show()
