from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


@dataclass
class SubPlot:
    """Create single subplot of the results plot: add data and format axes"""
    plot: str  # TODO: set type
    title: str
    x_data: np.array
    y_data: np.array
    label: str
    y_axis_format: str
    y_axis_decimal_places: int = 0

    def __post_init__(self):
        # plot data
        self.plot.plot(self.x_data, self.y_data, label=self.label)

        # set y axis format
        if self.y_axis_format == "percent":
            self.plot.yaxis.set_major_formatter(
                ticker.PercentFormatter(decimals=self.y_axis_decimal_places, xmax=1)
            )
        if self.y_axis_format == "dollar":
            self.plot.yaxis.set_major_formatter("${x:1.0f}")

        # add title
        self.plot.set_title(self.title)


def plot_results(models: dict, plot_to_year: int = None):
    """Returns a figure plotting the results of DICE model run(s). 
    
    Inputs:
        models - A dictionary of {model: "model name"}
    
    Outputs:
        fig - A figure containing plots of each model's results, labelled with the model name"""

    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams.update({"font.size": 9})

    # set up figure and suplots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3)

    SCC_plot = fig.add_subplot(gs[0, 0:2])
    S_plot = fig.add_subplot(gs[0, 2])
    Lambda_plot = fig.add_subplot(gs[1, 0])
    mu_plot = fig.add_subplot(gs[1, 1])
    Omega_plot = fig.add_subplot(gs[2, 2])
    E_plot = fig.add_subplot(gs[1, 2])
    E_cum_plot = fig.add_subplot(gs[2, 0])
    T_plot = fig.add_subplot(gs[2, 1])

    for model, model_name in models.items():
        SubPlot(
            plot=SCC_plot,
            title=r"SCC - $/tCO2",
            x_data=model.setup.T,
            y_data=model.SCC,
            label=model_name,
            y_axis_format="dollar",
        )

        SubPlot(
            plot=S_plot,
            title=r"Savings rate",
            x_data=model.setup.T,
            y_data=model.optimization.result_control_variables.S,
            label=model_name,
            y_axis_format="percent",
            y_axis_decimal_places=1,
        )

        SubPlot(
            plot=Lambda_plot,
            title=r"Abatement costs ($\Lambda$) - % GDP",
            x_data=model.setup.T,
            y_data=model.result_calc.Lambda,
            label=model_name,
            y_axis_format="percent",
            y_axis_decimal_places=1,
        )

        SubPlot(
            plot=mu_plot,
            title=r"Emissions control rate ($\mu$)",
            x_data=model.setup.T,
            y_data=model.optimization.result_control_variables.mu,
            label=model_name,
            y_axis_format="percent",
        )

        SubPlot(
            plot=Omega_plot,
            title=r"Damages - % GDP",
            x_data=model.setup.T,
            y_data=1 - (1 / (1 + model.result_calc.Omega)),
            label=model_name,
            y_axis_format="percent",
            y_axis_decimal_places=1,
        )

        SubPlot(
            plot=E_plot,
            title=r"Emissions - GtCO2/year",
            x_data=model.setup.T,
            y_data=model.result_calc.E,
            label=model_name,
            y_axis_format="number",
        )

        SubPlot(
            plot=E_cum_plot,
            title=r"Cumulative emissions - GtCO2",
            x_data=model.setup.T,
            y_data=model.result_calc.E.cumsum() * model.parameters.timestep,
            label=model_name,
            y_axis_format="number",
        )

        SubPlot(
            plot=T_plot,
            title=r"Temperature - $\degree$C above 1900",
            x_data=model.setup.T,
            y_data=model.result_calc.T_AT,
            label=model_name,
            y_axis_format="number",
        )

    for _, ax in enumerate(fig.axes):
        if len(models) > 1:
            ax.legend()

        if plot_to_year:
            ax.set_xlim(xmin=min([model.setup.T[0] for model in models.keys()]))
            ax.set_xlim(xmax=plot_to_year)

    return fig
