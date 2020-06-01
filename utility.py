
def main():
    fileName = "2M_PCA"
    with open("Output/" + fileName, "r", encoding="utf-8") as inputFile:
        data = inputFile.read().split("\n")

    with open("Output/Losses_Rewards_" + fileName + ".csv", "w", encoding="utf-8") as outputFile:
        outputFile.write("losses, rewards\n")
        for index in range(len(data)):
            outputFile.write(data[index].split(" ")[-1] + ("\n" if index % 2 else ", "))
        outputFile.close()
    return


if __name__ == '__main__':
    main()
