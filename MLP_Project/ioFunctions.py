def writeStats(fileName, stat):
    with open(fileName, 'w') as file:
        file.write(str(stat))
        file.write('\n')
