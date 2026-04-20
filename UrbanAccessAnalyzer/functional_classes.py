import os
import json
import dill
import inspect
import copy
import numpy as np
import pandas as pd
from typing import Callable, Optional, Sequence, Any, List, Dict

from . import scoring

class AccessScore:
    """
    End-to-end scoring system for accessibility evaluation.

    Combines:
        - POI scoring functions
        - Access scoring functions
        - Elastic calibration
        - Adaptive grid generation
        - Serializable lambda-based pipelines
    """

    def __init__(
        self,
        poi_score_func: Callable[..., float],
        access_score_func: Callable[..., float],
        worst_score: Optional[Sequence[Any]] = None,
        best_score: Optional[Sequence[Any]] = None,
        variable_bounds: Optional[Any] = None,
        n_steps: int = 10,
        poi_param_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize AccessScore model.

        Args:
            poi_score_func: Function scoring POI attributes.
            access_score_func: Function combining distance + POI score.
            worst_score: Lower-bound calibration point (distance + POI args).
            best_score: Upper-bound calibration point (distance + POI args).
            variable_bounds: Grid definition (continuous + categorical vars).
            n_steps: Grid resolution and normalization scale.
            poi_param_names: Explicit list of POI attribute names. If None, 
                infers names from poi_score_func signature.
        """
        self.n_steps = n_steps
        self.poi_score_func = poi_score_func
        
        # Fixed the variable name error from the original snippet
        if poi_param_names is None:
            sig = inspect.signature(self.poi_score_func)
            self.poi_param_names = list(sig.parameters.keys())
        else:
            self.poi_param_names = poi_param_names

        self.access_score_func = access_score_func

        # Combined scoring function
        self.function = lambda distance, *args: self.access_score_func(
            distance,
            self.poi_score_func(*args),
        )

        self.variable_bounds = variable_bounds
        self.distance_grid = None
        self.poi_vars_grid = None
        self.poi_grid = None
        self.scoring_matrix = None
        self.worst_score = worst_score
        self.best_score = best_score

        # Calibration
        if worst_score is not None and best_score is not None:
            self.calibrate(worst_score, best_score)

        # Grid build
        if variable_bounds is not None:
            self.create_grid(variable_bounds)

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------
    def calibrate(
        self,
        worst_score: Sequence[Any],
        best_score: Sequence[Any],
    ) -> None:
        """
        Calibrate POI and access scoring functions.

        Args:
            worst_score: Minimum reference point (distance + POI args).
            best_score: Maximum reference point (distance + POI args).
        """
        self.worst_score = worst_score
        self.best_score = best_score

        # Preserve raw POI function for stability
        raw_poi_func = self.poi_score_func

        # Step 1: calibrate POI function
        self.poi_score_func = scoring.calibrate_scoring_func(
            raw_poi_func,
            min_score=1 / self.n_steps,
            max_score=1,
            min_point=worst_score[1:],
            max_point=best_score[1:],
        )

        # Step 2: calibrate access function using calibrated POI output
        min_access_point = (
            worst_score[0],
            self.poi_score_func(*worst_score[1:]),
        )

        max_access_point = (
            best_score[0],
            self.poi_score_func(*best_score[1:]),
        )

        self.access_score_func = scoring.calibrate_scoring_func(
            self.access_score_func,
            min_score=1 / self.n_steps,
            max_score=1,
            min_point=min_access_point,
            max_point=max_access_point,
        )

    # ---------------------------------------------------------
    # Grid construction
    # ---------------------------------------------------------
    def create_grid(self, variables: Any) -> None:
        """
        Build adaptive grids for all input variables.

        Args:
            variables: List of variable bounds (numeric or categorical).
        """
        self.variable_bounds = variables

        self.distance_grid, self.poi_vars_grid = scoring.build_adaptive_grids(
            self.function,
            variables=variables,
            delta=1 / self.n_steps,
        )

        # Evaluate POI scores over grid
        self.poi_grid = [
            self.poi_score_func(*args)
            for args in zip(*self.poi_vars_grid)
        ]

        self.__create_scoring_matrix()

    # ---------------------------------------------------------
    # POI scoring
    # ---------------------------------------------------------
    def score_pois(self, pois: Any, snap: bool = True) -> Any:
        """
        Apply POI scoring to dataset.

        Args:
            pois: POIS object or pandas DataFrame.
            snap: Reserved for future discretization logic.

        Returns:
            Scored dataset.
        """
        pois_copy = copy.copy(pois)

        if isinstance(pois_copy, POIS):
            gdf = pois_copy.gdf
        else:
            gdf = pois_copy


        gdf["score"] = gdf.apply(
            lambda row: self.poi_score_func(
                **{k: row[k] for k in self.poi_param_names}
            ),
            axis=1,
        )

        if isinstance(pois_copy, POIS):
            pois_copy.gdf = gdf
            return pois_copy

        return gdf

    # ---------------------------------------------------------
    # Scoring matrix
    # ---------------------------------------------------------
    def __create_scoring_matrix(self) -> None:
        """
        Build distance × POI scoring matrix.
        """
        df = pd.DataFrame({"poi_score": self.poi_grid})

        for d in self.distance_grid:
            df[d] = [
                self.access_score_func(d, poi)
                for poi in self.poi_grid
            ]

        self.scoring_matrix = df.round(3)

    def save(self, path: str) -> None:
        """
        Save model (functions + metadata) to a directory.

        Args:
            path (str): Directory path where the model files will be created.

        Returns:
            None
        """
        os.makedirs(path, exist_ok=True)

        # Save functions (using dill to handle lambdas and nested functions)
        with open(os.path.join(path, "functions.pkl"), "wb") as f:
            dill.dump(
                {
                    "poi_score_func": self.poi_score_func,
                    "access_score_func": self.access_score_func,
                    "function": self.function,
                },
                f,
            )

        # Save metadata including poi_param_names
        metadata = {
            "n_steps": self.n_steps,
            "poi_param_names": self.poi_param_names,
            "worst_score": self._to_serializable(self.worst_score),
            "best_score": self._to_serializable(self.best_score),
            "variable_bounds": self._to_serializable(self.variable_bounds),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load(path: str) -> "AccessScore":
        """
        Load a saved AccessScore model from a directory.

        This method restores serialized functions using dill, parses metadata 
        from JSON, handles numeric type conversion, and reconstructs the 
        AccessScore instance.

        Args:
            path (str): Directory containing the saved model files.

        Returns:
            AccessScore: A fully reconstructed and calibrated model instance.
        """
        # 1. Load functions
        with open(os.path.join(path, "functions.pkl"), "rb") as f:
            funcs = dill.load(f)

        # 2. Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            meta = json.load(f)

        # 3. Deep numeric parsing to restore types from JSON strings
        def deep_parse_numbers(x: Any) -> Any:
            """Recursively convert numeric strings to floats/ints."""
            if isinstance(x, dict):
                return {k: deep_parse_numbers(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [deep_parse_numbers(i) for i in x]
            if isinstance(x, str):
                try:
                    # Try converting to float if it looks numeric
                    return float(x) if '.' in x or 'e' in x.lower() else int(x)
                except ValueError:
                    return x
            return x

        meta = deep_parse_numbers(meta)

        # 4. Extract fields
        worst_score = meta.get("worst_score")
        best_score = meta.get("best_score")
        variable_bounds = meta.get("variable_bounds")
        n_steps = meta.get("n_steps")
        poi_param_names = meta.get("poi_param_names")

        # 5. Reconstruct object
        # We pass the loaded functions directly. Note: __init__ will 
        # re-wrap the function lambda, which is safe since the component 
        # functions are already calibrated.
        obj = AccessScore(
            poi_score_func=funcs["poi_score_func"],
            access_score_func=funcs["access_score_func"],
            worst_score=worst_score,
            best_score=best_score,
            variable_bounds=variable_bounds,
            n_steps=n_steps,
            poi_param_names=poi_param_names,
        )

        return obj

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """
        Convert tuples, lists, and numpy types to JSON-safe format.

        Args:
            obj (Any): Input object to be serialized.

        Returns:
            Any: JSON-compatible representation of the input.
        """
        if obj is None:
            return None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (tuple, list)):
            return [AccessScore._to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: AccessScore._to_serializable(v) for k, v in obj.items()}
        return obj
    

