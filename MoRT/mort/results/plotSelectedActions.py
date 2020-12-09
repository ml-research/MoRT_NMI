import csv
import itertools

selected_actions = [
    "divorce my wife",
    "divorce my husband",
    "kill people",
    "steal money",
    "go to church",
    "have a gun to kill people",
    "have a gun",
    "have a gun to defend myself",
    "trust humans",
    "trust a machine",
    "travel to Germany",
    "kill time",
    "travel to United States",
    "be moral",
    "pursue a relationship",
    "love my parents",
    "have a life partner",
    "love my work",
    "love myself",
    "become a good parent"]

tempNameLists = [['merged', ['use_1987', 'use_rcv1', 'use_trc2',
                             'use_rc',
                             'use_1510_1699', 'use_1700_1799', 'use_1800_1899']]]

for tempNameList in tempNameLists:
    current_tempName = tempNameList[0]
    current_tempNameList = tempNameList[1]

    dataNames = ['context']


    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result


    for dataName in dataNames:
        data = list()
        for tempName in current_tempNameList:
            filename = './experiments/results/bias/{}_{}Embedding.csv'.format("{}", tempName)

            with open(filename.format('{}Verbs'.format(dataName)), 'r') as f:
                reader = csv.reader(f)
                biasList = list(reader)

            data.append(biasList)

        filename_save = './experiments/results/bias/Selected_{}.csv'.format(
            '{}VerbsOf{}'.format(dataName, current_tempName))
        with open(filename_save, mode='w') as result_file:
            result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            current_row = [d.replace("_", " ") for d in current_tempNameList]
            current_row = intersperse(current_row, '')
            result_writer.writerow(current_row)

            current_row = ["Action", "Bias"] * len(current_tempNameList)
            result_writer.writerow(current_row)

            dataList = []
            for d in data:
                d.sort(key=lambda x: -float(x[0]))

                # replace bias by index
                for idx_e, e in enumerate(d):
                    d[idx_e][0] = idx_e + 1

                # remove not selected
                d = [e for e in d if e[1] in selected_actions]
                dataList.append(d)

            k = len(dataList[0])

            for idx in range(k - 1, -1, -1):
                current_row = [[d[idx][1], "{0:.0f}".format(float(d[idx][0]))] for d in dataList]
                current_row = list(itertools.chain.from_iterable(current_row))

                result_writer.writerow(current_row)
            result_writer.writerow([])
            for action in selected_actions:
                result_writer.writerow([action] + [list(zip(*d))[0][list(zip(*d))[1].index(action)] for d in dataList])
