import os
import pickle
from matplotlib import pyplot as plt

from elevator_rl.environment.episode_summary import combine_summaries


def main():
    dir_name = "saved_summaries2/"
    directory = os.fsencode(dir_name)

    summarydict = {
        i: {j: {k: [] for k in [0, 10, 20, 50, 100, 200]} for j in [1, 2]}
        for i in range(3, 11)
    }

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(dir_name + filename, "rb") as handle:
            current_summaries = pickle.load(handle)

        if "mcts0_" in filename:
            mcts = 0
        elif "mcts10_" in filename:
            mcts = 10
        elif "mcts20_" in filename:
            mcts = 20
        elif "mcts50_" in filename:
            mcts = 50
        elif "mcts100_" in filename:
            mcts = 100
        elif "mcts200_" in filename:
            mcts = 200
        else:
            raise ValueError("unknown file name format: " + filename)

        if "floors3_" in filename:
            floors = 3
        elif "floors4_" in filename:
            floors = 4
        elif "floors5_" in filename:
            floors = 5
        elif "floors6_" in filename:
            floors = 6
        elif "floors7_" in filename:
            floors = 7
        elif "floors8_" in filename:
            floors = 8
        elif "floors9_" in filename:
            floors = 9
        elif "floors10_" in filename:
            floors = 10
        else:
            raise ValueError("unknown file name format: " + filename)

        if "elevs1" in filename:
            elevs = 1
        elif "elevs2" in filename:
            elevs = 2
        else:
            raise ValueError("unknown file name format: " + filename)
        # print(current_summaries[0])

        summarydict[floors][elevs][mcts] = current_summaries
    print("done reading")

    plot(
        summarydict, 1, lambda x: x.percent_transported(), "% of passengers transported"
    )

    plot(summarydict, 1, lambda x: x.quadratic_waiting_time, "quadratic waiting time")


def plot(summarydict, elevs, value_extractor, y_axis):
    plt.clf()
    for mcts in [0, 20, 200]:
        x, y = [], []
        for floors in range(3, 9):
            x.append(floors)
            summaries = summarydict[floors][elevs][mcts]
            avg, stddev = combine_summaries(summaries)
            y.append(value_extractor(avg))
        if mcts == 0:
            label = "Random Policy"
            plt.plot(x, y, label=label, linewidth=3)
        else:
            label = f"mcts samples: {mcts}"
            plt.plot(x, y, label=label)
    axes = plt.gca()
    axes.set_xlabel("nr of floors")
    axes.set_ylabel(y_axis)
    plt.title(f"houses with {elevs} elevator(s)")
    plt.legend()
    plt.savefig(f"{elevs}_elevs_{y_axis}.png",dpi=400)


if __name__ == "__main__":
    main()
