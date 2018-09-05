import pandas as pd
import matplotlib.pyplot as plt

import topic_model
import labels_to_documents

def map_range(x, ab, cd=(0,1)):
    y = (x-ab[0]) * (cd[1]-cd[0])/(ab[1]-ab[0]) + cd[0]
    return y

def circle_plot(perp):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.set_title("Alpha/Beta Grid Search\n"
                 + "Min Radius: perplexity = {0:.2f}\n".format(perp['perplexity'].min())
                 + "Max Radius: perplexity = {0:.2f}\n".format(perp['perplexity'].max()))

    ax.set_xlabel("alpha")
    xlocs = [i + 1 for i in range(len(alpha))]
    ax.set_xticks(xlocs)
    ax.set_xticklabels(alpha)
    ax.set_xlim((0, len(alpha) + 1))

    ax.set_ylabel("beta")
    ylocs = [i + 1 for i in range(len(beta))]
    ax.set_yticks(ylocs)
    ax.set_yticklabels(beta)
    ax.set_ylim((0, len(beta) + 1))

    radius_range = (0.01, 0.5)
    perplexity_range = (perp['perplexity'].min(), perp['perplexity'].max())

    for i in range(len(perp)):
        r = map_range(perp['perplexity'][i], perplexity_range, radius_range)
        print(r)
        print((alpha.index(perp['alpha'][i]) + 1, beta.index(perp['beta'][i]) + 1))
        circle = plt.Circle(xy=(alpha.index(perp['alpha'][i]) + 1, beta.index(perp['beta'][i]) + 1),
                            radius=r, color='b')
        ax.add_artist(circle)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    path = "./out/model/"

    labels_to_documents.main()

    alpha = [0.9, 0.5, 0.1, 0.01, 0.001, 0.0001]
    beta = [0.9, 0.5, 0.1, 0.01, 0.001, 0.0001]
    num_topics = [5, 10, 20, 50, 100]

    perplexity_df = pd.DataFrame(columns=['alpha', 'beta', 'num_topics''perplexity'])

    for a in alpha:
        for t in num_topics:
            print("RUNNING TOPIC MODEL WITH alpha=%f, beta=%f, T=%d" % (a, 0.001, t))
            topic_model.main(alpha=a, beta=0.001, T=t, W=10000)
            df = pd.read_csv(filepath_or_buffer=path+"perplexity.csv",
                                 header=None)
            p = df.sum()[1]
            perplexity_df = perplexity_df.append({'alpha':a, 'beta':0.001, 'num_topics':t,
                                                      'perplexity':p}, ignore_index=True)

    for b in beta:
        for t in num_topics:
            print("RUNNING TOPIC MODEL WITH alpha=%f, beta=%f, T=%d" % (0.001, b, t))
            topic_model.main(alpha=0.001, beta=b, T=t, W=10000)
            df = pd.read_csv(filepath_or_buffer=path+"perplexity.csv",
                                 header=None)
            p = df.sum()[1]
            perplexity_df = perplexity_df.append({'alpha':0.001, 'beta':b, 'num_topics':t,
                                                      'perplexity':p}, ignore_index=True)

    print("Done.")
    perplexity_df.to_csv("/Users/bergamaschi/PycharmProjects/hb-song-analysis/alpha_beta_topic_sweep_2.csv")



