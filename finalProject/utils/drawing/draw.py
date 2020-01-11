import cv2
import matplotlib.pyplot as plt
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.drawing.common import draw_str
from finalProject.utils.matchers.Matchers import kaze_matcher, flannmatcher


def DrawOnFrameMyIds(myids, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thicknessText = 1

    colorBlue = (255, 0, 0)
    colorRed = (0, 0, 255)

    radius = 3
    thicknessCircle = -1
    thicknessRec = 2

    for _id in myids.values():
        frame = cv2.rectangle(frame, myids[_id["_id"]]["box"][0], myids[_id["_id"]]["box"][1],
                              colorBlue, thicknessRec)
        frame = cv2.circle(frame, myids[_id["_id"]]["centerHuman"], radius, colorRed, thicknessCircle)
        frame = cv2.putText(frame, 'ID:' + str(_id["_id"]), (myids[_id["_id"]]["centerHuman"][0]
                                                             , myids[_id["_id"]]["centerHuman"][1] - 50), font,
                            fontScale, (255, 0, 0), thicknessText, cv2.LINE_AA)
    return frame


def DrawHumans(MyPeople, frame, affectedPeople):
    thicknessRec = 2
    for index in affectedPeople:
        color = MyPeople[index].colorIndex
        # print(color)
        cv2.rectangle(frame, MyPeople[index].locations[-1][0], MyPeople[index].locations[-1][1],
                      color, thicknessRec)
        draw_str(frame, MyPeople[index].locations[-1][0], "id " + str(MyPeople[index].indexCount))


def DrawSource(mySource, frame):
    thicknessRec = 2
    color = (255, 100, 150)
    cv2.rectangle(frame, mySource.locations[-1][0], mySource.locations[-1][1],
                  color, thicknessRec)
    draw_str(frame, mySource.locations[-1][0], "id " + str(mySource.indexCount))


def ShowPeopleTable(MyPeople, config: "configFile"):
    if len(MyPeople) == 0:
        print("no people were found!")
    else:
        if config["showHistory"]:
            photos = "history"
        else:
            photos = "frames"

        maxFramesHuman = max(MyPeople, key=lambda human: len(human.__getattribute__(photos)))

        rows = len(list(filter(lambda human: len(human.__getattribute__(photos)) > 0, MyPeople))) + 1

        cols = len(maxFramesHuman.__getattribute__(photos)) + 1

        print("rows ", rows)
        print("cols", cols)

        if rows > 0 and cols > 0:
            fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
            for idx, human in enumerate(MyPeople):
                for jdx, frame in enumerate(human.__getattribute__(photos)):
                    print(idx, jdx)
                    ax[idx, jdx].imshow(frame)

            plt.show()


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        # convert BGR to RGB
        # images[i] = cv2.cvtColor(images[i],images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(images[i])
    plt.show(block=True)


def drawOnScatter(ax, keyPoints, color, label="none"):
    xyList = list(map(lambda keypoint: keypoint.pt, keyPoints))
    xl, yl = zip(*xyList)
    scale = 10
    ax.scatter(xl, yl, c=color, s=scale, label=label,
               alpha=0.8, edgecolors='none')


def drawFramePair(frameObjectSource, frameObjectTarget, NameAlgo, ax, options):
    frameObjectSource["frame"] = cv2.cvtColor(frameObjectSource["frame"], cv2.COLOR_BGR2RGB)
    frameObjectTarget["frame"] = cv2.cvtColor(frameObjectTarget["frame"], cv2.COLOR_BGR2RGB)

    if NameAlgo in ["KAZE", "ORB"]:
        matches = kaze_matcher(frameObjectSource[NameAlgo]["des"],
                               frameObjectTarget[NameAlgo]["des"])
    else:
        matches = flannmatcher(frameObjectSource[NameAlgo]["des"],
                               frameObjectTarget[NameAlgo]["des"])

    out_img = np.array([])

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=0)

    out_img = cv2.drawMatchesKnn(frameObjectSource["frame"], frameObjectSource[NameAlgo]["keys"],
                                 frameObjectTarget["frame"]
                                 , frameObjectTarget[NameAlgo]["keys"],
                                 matches[:options["max_matches"]], **draw_params, outImg=None)

    ax.imshow(out_img)

    ax.set_xlabel("algorithm name  {} \n Number Of matches {}  from {} keys points"
                  .format(NameAlgo, len(matches),
                          len(frameObjectTarget[NameAlgo]["keys"])
                          + len(frameObjectSource[NameAlgo]["keys"])))

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.grid(True)

    # return ax
    # plt.show()


def drawTargetFinal(acc_targets, options):
    acc_targets = sorted(acc_targets.items(), key=lambda item: item[1]["maxAcc"])
    most_acc_target = acc_targets[0][1]

    algoritamDraw = [algo.name for algo in NamesAlgorithms]

    fig, axes = plt.subplots(nrows=len(algoritamDraw), figsize=(15, 15))

    for index, algoName in enumerate(algoritamDraw):
        drawFramePair(most_acc_target["frameSource"],
                      most_acc_target["frameTarget"],
                      algoName, axes[index], options)

    axes[-1].set_xlabel(str(axes[-1].get_xlabel()) +
                        "\n \n" + "final matches factor : " + str(most_acc_target["maxAcc"]))

    plt.show()
