import sys
import pytest

sys.path.append("../PythonDICE")

from DICE import DICE
from PlotResults import plot_results


@pytest.fixture
def test_parameters():
    return {"max_iterations": 2, "n_timesteps": 3}


@pytest.fixture
def test_model(test_parameters):
    return DICE(user_parameters=test_parameters)


def test_parameters_update(test_model):
    assert len(test_model.result_calc.T_AT) == 3
    assert test_model.setup.T[0] == 2010


def test_no_optimization(test_parameters):
    m = DICE(optimize=False, user_parameters=test_parameters)
    with pytest.raises(Exception):
        m.result_calc

    m.run_optimization()
    assert len(m.result_calc.T_AT) == 3


def test_use_updated_parameters(test_parameters):
    m = DICE(use_updated_parameters=True, user_parameters=test_parameters)
    assert m.parameters.start_year == 2025


def test_plot_single_model_results(test_model):
    fig = test_model.plot_results()
    fig.savefig("tests/test_fig.png", bbox_inches="tight")

def test_plot_to_year(test_model):
    fig = test_model.plot_results(plot_to_year = 2015)
    fig.savefig("tests/test_fig2.png", bbox_inches="tight")


def test_plot_multiple_models(test_model, test_parameters):
    m2 = DICE(use_updated_parameters=True, user_parameters=test_parameters)
    fig = plot_results({test_model: "test 1", m2: "test 2"})
    fig.savefig("tests/test_fig3.png", bbox_inches="tight")
