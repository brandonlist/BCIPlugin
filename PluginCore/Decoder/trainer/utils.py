import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def return_df_search(searcher, keys):
    data = searcher.cv_results_
    metrics = []
    for key in keys:
        metrics.append('param_' + key)
    metrics.append('mean_test_score')
    df = pd.DataFrame({key: value for (key, value) in data.items() if key in metrics})
    return df


def boxplot_res(df, interest_keys, gs=None):
    if df is None:
        assert gs is not None
        df = return_df_search(gs, interest_keys)
    if len(interest_keys) == 1:
        sns.boxplot(x='param_' + interest_keys[0], y='mean_test_score', data=df)
    else:
        sns.boxplot(x='param_' + interest_keys[0], y='mean_test_score', hue='param_' + interest_keys[1], data=df)


def plot_res( gs, interest_key):
    df = return_df_search(gs, [interest_key])
    plt.plot(df['param_' + interest_key], df['mean_test_score'], '.')


def param3d_viz(gs, params, score_name='mean_test_score'):
    df = return_df_search(gs, params)

    scores = df[score_name].to_numpy()
    v_min = scores.min()
    v_max = scores.max()

    param_0 = df['param_' + params[0]].to_numpy()
    param_1 = df['param_' + params[1]].to_numpy()
    param_2 = df['param_' + params[2]].to_numpy()

    res = np.vstack([param_0, param_1, param_2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3d图需要加projection='3d'
    ax.scatter(res[0, :], res[1, :], res[2, :], c=scores, cmap='plasma')

    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=score_name, orientation='horizontal')

    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_zlabel(params[2])