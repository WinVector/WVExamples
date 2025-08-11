import numpy as np
import pandas as pd
from plotnine import *


n = 100
error_scale = 0.4

# different than default seed, so this rng is not guessing other rng's state
v_true = np.random.default_rng(3762).normal(size=n)


def estimate_v(rng):
    return v_true + error_scale * rng.normal(size=n)


value_order = [
    "self evaluation (common method)",
    "ideal evaluation (not implementable)",
    "hold out evaluation (implementable)",
]
color_map = {
    "hold out evaluation (implementable)": "#1b9e77",
    "ideal evaluation (not implementable)": "#d95f02",
    "self evaluation (common method)": "#7570b3",
}
linetype_map = {
    "hold out evaluation (implementable)": "dashed",
    "ideal evaluation (not implementable)": "dotted",
    "self evaluation (common method)": "solid",
}


def plot_1(d: pd.DataFrame, *, d_max: pd.DataFrame, title: str):
    return (
        ggplot()
        + geom_line(
            data=d,
            mapping=aes(x="step number", y="self evaluation (common method)"),
            color=color_map["self evaluation (common method)"],
            size=1,
        )
        + geom_point(
            data=d_max,
            mapping=aes(x="step number", y="self evaluation (common method)"),
            size=5,
            color=color_map["self evaluation (common method)"],
        )
        + geom_hline(yintercept=0, alpha=0.5)
        + geom_text(
            data=d_max,
            mapping=aes(
                x="step number",
                y="self evaluation (common method)",
                label="self evaluation (common method)",
            ),
            nudge_y=2,
            nudge_x=3,
            format_string="${:.2f}",
            color=color_map["self evaluation (common method)"],
        )
        + geom_vline(
            mapping=aes(xintercept="step number"),
            data=d_max,
            linetype=linetype_map["self evaluation (common method)"],
            color=color_map["self evaluation (common method)"],
            size=2,
        )
        + theme(legend_position="bottom", legend_key_height=50, legend_key_width=50)
        + ggtitle(title)
    )


def plot_k(
    d: pd.DataFrame,
    *,
    title: str,
    observed_values: bool = True,
    ideal_values: bool = True,
):
    d = d.copy()
    del d["x"]
    d_plot = d.melt(id_vars=["step number"], var_name="evaluation strategy")
    vs = [v for v in value_order if v in set(d_plot["evaluation strategy"])]
    d_plot["evaluation strategy"] = pd.Categorical(
        d_plot["evaluation strategy"], categories=vs, ordered=True
    )
    d_plot_maxes = d_plot.loc[
        d_plot.groupby("evaluation strategy", observed=True)["value"].idxmax(), :
    ]
    d_stopping_values = d_plot.loc[
        np.isin(d_plot["step number"], list(set(d_plot_maxes["step number"])))
        & (d_plot["evaluation strategy"] == "ideal evaluation (not implementable)"),
        :,
    ]
    plt = (
        ggplot()
        + geom_hline(yintercept=0, alpha=0.5)
        + geom_line(
            data=d_plot,
            mapping=aes(
                x="step number",
                y="value",
                color="evaluation strategy",
                linetype="evaluation strategy",
            ),
            size=1,
        )
        + geom_point(
            data=d_plot_maxes,
            mapping=aes(x="step number", y="value", color="evaluation strategy"),
            size=5,
        )
        + geom_vline(
            mapping=aes(
                xintercept="step number",
                color="evaluation strategy",
                linetype="evaluation strategy",
            ),
            data=d_plot_maxes,
            size=2,
        )
        + geom_point(
            data=d_stopping_values,
            mapping=aes(x="step number", y="value"),
            alpha=0.5,
            size=3,
        )
        + theme(legend_position="bottom", legend_key_height=50, legend_key_width=50)
        + scale_color_manual(values=color_map)
        + scale_linetype_manual(values=linetype_map)
        + ggtitle(title)
    )
    if observed_values:
        plt = plt + geom_text(
            data=d_plot_maxes,
            mapping=aes(
                x="step number", y="value", label="value", color="evaluation strategy"
            ),
            nudge_y=2,
            nudge_x=3,
            format_string="${:.2f}",
        )
    if ideal_values:
        plt = plt + geom_text(
            data=d_stopping_values,
            mapping=aes(x="step number", y="value", label="value"),
            nudge_y=2,
            nudge_x=3,
            format_string="${:.2f}",
        )
    d_stopping_values = d_stopping_values.rename(
        columns={"value": "actual value"}, inplace=False
    )
    d_plot_maxes = d_plot_maxes.rename(
        columns={
            "evaluation strategy": "stopping strategy",
            "value": "self reported value",
        },
        inplace=False,
    )
    d_summary = pd.merge(
        d_plot_maxes,
        d_stopping_values[["step number", "actual value"]],
        on=["step number"],
        how="left",
    ).sort_values(["stopping strategy"], ignore_index=True, inplace=False)
    return (
        plt,
        d_summary,
    )


def plt_d1(
    d_density: pd.DataFrame,
    *,
    initial_self_eval_error: float,
    initial_out_of_sample_error: float,
):
    v_frame = pd.DataFrame(
        {
            "measure": [
                "v_approx (used for optimization)",
                "v_approx_2 (used for evaluation and stopping)",
            ],
            "estimated error": [initial_self_eval_error, initial_out_of_sample_error],
        }
    )
    cm = {
        "v_approx (used for optimization)": color_map[
            "self evaluation (common method)"
        ],
        "v_approx_2 (used for evaluation and stopping)": color_map[
            "hold out evaluation (implementable)"
        ],
    }
    lm = {
        "v_approx (used for optimization)": linetype_map[
            "self evaluation (common method)"
        ],
        "v_approx_2 (used for evaluation and stopping)": linetype_map[
            "hold out evaluation (implementable)"
        ],
    }
    return (
        ggplot(data=d_density, mapping=aes(x="estimated error"))
        + geom_density(fill="lightgrey", alpha=0.7)
        + geom_vline(xintercept=0, alpha=0.5)
        + geom_vline(
            data=v_frame,
            mapping=aes(
                xintercept="estimated error", color="measure", linetype="measure"
            ),
            size=2,
        )
        + scale_color_manual(values=cm)
        + scale_linetype_manual(values=lm)
        + theme(legend_position="bottom", legend_key_height=50, legend_key_width=50)
        + ggtitle("distribution of objective errors on random x")
    )


def plt_d2(
    d_density,
    d_density_opt,
    *,
    initial_self_eval_error: float,
    naive_self_eval_error: float,
    initial_out_of_sample_error: float,
    final_out_of_sample_error: float,
):
    v_frame = pd.DataFrame(
        {
            "situation": [
                "before optimization",
                "before optimization",
                "post optimization",
                "post optimization",
            ],
            "measure": [
                "v_approx (used for optimization)",
                "v_approx_2 (used for evaluation and stopping)",
                "v_approx (used for optimization)",
                "v_approx_2 (used for evaluation and stopping)",
            ],
            "estimated error": [
                initial_self_eval_error,
                initial_out_of_sample_error,
                naive_self_eval_error,
                final_out_of_sample_error,
            ],
        }
    )
    cm = {
        "v_approx (used for optimization)": color_map[
            "self evaluation (common method)"
        ],
        "v_approx_2 (used for evaluation and stopping)": color_map[
            "hold out evaluation (implementable)"
        ],
    }
    lm = {
        "v_approx (used for optimization)": linetype_map[
            "self evaluation (common method)"
        ],
        "v_approx_2 (used for evaluation and stopping)": linetype_map[
            "hold out evaluation (implementable)"
        ],
    }
    return (
        ggplot(
            data=pd.concat([d_density, d_density_opt]), mapping=aes(x="estimated error")
        )
        + geom_density(fill="lightgrey", alpha=0.7)
        + geom_vline(xintercept=0, alpha=0.5)
        + geom_vline(
            data=v_frame,
            mapping=aes(
                xintercept="estimated error", color="measure", linetype="measure"
            ),
            size=2,
        )
        + scale_color_manual(values=cm)
        + scale_linetype_manual(values=lm)
        + facet_wrap("situation", ncol=1, scales="free_y")
        + theme(legend_position="bottom", legend_key_height=50, legend_key_width=50)
        + ggtitle("distribution of objective errors for optimized x")
    )
