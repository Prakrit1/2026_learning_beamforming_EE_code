
import numpy as np
import matplotlib.pyplot as plt
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling, change_lightness,
)


# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     # 'font.size': 10,
#     # 'axes.labelsize': 10,
#     # 'legend.fontsize': 10,
#     # 'xtick.labelsize': 10,
#     # 'ytick.labelsize': 10,
# })
from matplotlib import colors as mpl_colors

import src
from src.data.channel.get_steering_vec import get_steering_vec
from src.config.config_plotting import generic_styling


def plot_beampattern(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        user_manager: 'src.data.user_manager.UserManager',
        w_precoder: np.ndarray,
        w_mrc_precoder: np.ndarray,
        position_sweep_range: np.ndarray or None = None,
        plot_title: str or None = None,
        name: str = None,
        normalization: bool = True,
) -> None:
    """Plots beam power toward each user from the point of view of a satellite for a given precoding w_precoder."""
    plot_cfg = PlotConfig()
    # save original positions
    original_pos = [
        user.spherical_coordinates
        for user in user_manager.users
    ]

    # create a figure
    fig, ax = plt.subplots()
    line_styles = [ '-', 'dotted','-.']

    # mark user positions
    # for user in user_manager.users:
    #     ax.scatter(
    #         user.spherical_coordinates[2],
    #         0,
    #         color='gray',
    #         # label=f'user {user.idx}'
    #     )
        # for i, user in enumerate(user_manager.users):
        #     ax.axvline(
        #         user.spherical_coordinates[2],
        #         color='gray',
        #         # li
        #     )

    # calculate auto x axis scaling
    if position_sweep_range is None:
        max_dist = user_manager.users[-1].spherical_coordinates[2] - user_manager.users[0].spherical_coordinates[2]

        position_sweep_range = np.arange(
            user_manager.users[0].spherical_coordinates[2] - 0.5 * max_dist,
            user_manager.users[-1].spherical_coordinates[2] + 0.5 * max_dist,
            (user_manager.users[-1].spherical_coordinates[2] - user_manager.users[0].spherical_coordinates[2]) / 1000
        )

    # calculate power gains
    directional_power_gains = np.zeros((len(satellite_manager.satellites), len(user_manager.users), len(position_sweep_range)), dtype='complex128')
    directional_power_gains_mrc = np.zeros((len(satellite_manager.satellites), len(user_manager.users), len(position_sweep_range)), dtype='complex128')
    for position_id, position in enumerate(position_sweep_range):

        user_manager.users[0].update_position(
            [user_manager.users[0].spherical_coordinates[0], user_manager.users[0].spherical_coordinates[1], position]
        )
        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model,
                                                           user_manager=user_manager)

        for satellite in satellite_manager.satellites:
            # steering_vector_to_user = get_steering_vec(satellite=satellite,
            #                                            phase_aod_steering=np.cos(satellite.aods_to_users[0]))

            for user in user_manager.users:

                # directional_power_gain = np.matmul(steering_vector_to_user, w_precoder[satellite.idx:satellite.idx + satellite.antenna_nr, user.idx])
                directional_power_gain = np.matmul(satellite.channel_state_to_users[0], w_precoder[satellite.idx:satellite.idx + satellite.antenna_nr, user.idx])

                directional_power_gains[satellite.idx, user.idx, position_id] = directional_power_gain

                directional_power_gain_mrc = np.matmul(satellite.channel_state_to_users[0], w_mrc_precoder[satellite.idx:satellite.idx + satellite.antenna_nr, user.idx])

                directional_power_gains_mrc[satellite.idx, user.idx, position_id] = directional_power_gain_mrc


    sum_directional_power_gains = np.sum(directional_power_gains, axis=0)
    sum_directional_power_gains_mrc = np.sum(directional_power_gains_mrc, axis=0)
    sum_directional_power_gains = abs(sum_directional_power_gains)**2
    sum_directional_power_gains_mrc = abs(sum_directional_power_gains_mrc)**2


    if normalization:
        max_gain = np.max(sum_directional_power_gains)
        sum_directional_power_gains = sum_directional_power_gains/max_gain
        sum_directional_power_gains = sum_directional_power_gains[::-1]
        sum_directional_power_gains = 10*np.log10(sum_directional_power_gains)

        sum_directional_power_gains_mrc = sum_directional_power_gains_mrc[::-1]

    mrc_max_idx = np.argmax(sum_directional_power_gains_mrc, axis=1)
    mrc_max_positions = position_sweep_range[mrc_max_idx]

    # ax.plot(position_sweep_range, sum_directional_power_gains.T)

    # line_styles = ['dashed', '-.', 'dotted']
    # markers = ['o', 's', '^']
    generic_styling(ax=ax)

    for user_id in range(len(user_manager.users)):
        ax.plot(
            position_sweep_range,
            sum_directional_power_gains[user_id, :],
            color='black',
            linestyle=line_styles[user_id % len(line_styles)],
            # marker=markers[user_id % len(markers)],
            markevery=80,
            linewidth=2,
            label = rf'$\mathbf{{w}}_{{{user_id}}}^{{\mathrm{{{name}}}}}$'
        )

    ax.legend()
    ax.set_xlabel(r'Angle of Departure $\nu_{k}$ [deg]')
    if  normalization:
        ax.set_ylabel('Normalized Power Gain [dB]')
    else:
        ax.set_ylabel('Directional Power Gain')
    # if plot_title is not None:
    #     ax.set_title(plot_title)


    # ax.grid(False)


    # user_positions = [user.spherical_coordinates[2] for user in user_manager.users]
    user_positions = [pos[2] for pos in original_pos]
    user_labels = [rf'$\nu_{{{user.idx}}}$' for user in user_manager.users]

    mrc_labels = [rf'$\tilde{{\nu}}_{{{user.idx}}}$' for user in user_manager.users]
    mrc_max_positions = mrc_max_positions[::-1]


    all_positions = list(user_positions) + list(mrc_max_positions)
    all_labels = user_labels + mrc_labels

    user_positions = user_positions[::-1]

    # ax.set_xticks(user_positions)
    # ax.set_xticklabels(user_labels)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels)
    ax.grid(True, axis='x', which='major', color='gray', linestyle='--', linewidth=1.5, alpha=0.9)
    ax.legend(loc='lower right', framealpha=1)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)



    # ax.get_legend().remove()
    # from matplotlib.ticker import FuncFormatter
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{np.rad2deg(x):.1f}°'))
    # ax.set_xticks(np.deg2rad([80, 85, 90, 95, 100]))
    # ax.set_xticklabels(['80', '85', '90', '95', '100'])
    # reset original coordinates
    for user in user_manager.users:
        user.update_position(original_pos[user.idx])
        satellite_manager.calculate_satellite_distances_to_users(users=user_manager.users)
        satellite_manager.calculate_satellite_aods_to_users(users=user_manager.users)
        satellite_manager.update_channel_state_information(channel_model=config.channel_model,
                                                           user_manager=user_manager)


    # fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plot_cfg.plots_parent_path, plot_name='beampattern_'+name, padding=0.05)