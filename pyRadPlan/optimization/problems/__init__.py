"""Module containing treatment planning problem definitions."""

from ._optiprob import NonLinearPlanningProblem, PlanningProblem
from ._nonlin_fluence import NonLinearFluencePlanningProblem
from ._dao import DirectApertureOptimization
from ._factory import get_available_problems, get_problem, get_problem_from_pln, register_problem

register_problem(NonLinearFluencePlanningProblem)
register_problem(DirectApertureOptimization)

__all__ = [
    "NonLinearFluencePlanningProblem",
    "DirectApertureOptimization",
    "NonLinearPlanningProblem",
    "PlanningProblem",
    "get_available_problems",
    "get_problem",
    "get_problem_from_pln",
]
