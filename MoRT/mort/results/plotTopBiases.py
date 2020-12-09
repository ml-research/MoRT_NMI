import csv
import itertools

tempNameLists = [['decade', ['use_1800_1809', 'use_1810_1819', 'use_1820_1829', 'use_1830_1839', 'use_1840_1849',
                             'use_1850_1859', 'use_1860_1869', 'use_1870_1879', 'use_1880_1889', 'use_1890_1899']],
                 ['news', ['use_1987', 'use_rcv1', 'use_trc2']],
                 ['century', ['use_1510_1699', 'use_1700_1799', 'use_1800_1899']],
                 ['diff2', ['use_rc', 'use_1800_1899', 'use_trc2']]]

for tempNameList in tempNameLists:
    current_tempName = tempNameList[0]
    current_tempNameList = tempNameList[1]

    dataNames = ['context', 'atomic', 'allcorpora']


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

        filename_save = './experiments/results/bias/TopPos_{}.csv'.format(
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
                dataList.append(d)

            k = 25

            for idx in range(k):
                current_row = [[d[idx][1], "{0:.3f}".format(float(d[idx][0]))] for d in dataList]
                current_row = list(itertools.chain.from_iterable(current_row))

                result_writer.writerow(current_row)

        print('TopPos file has been saved.. to', filename_save)

        filename_save = './experiments/results/bias/TopNeg_{}.csv'.format(
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
                d.sort(key=lambda x: float(x[0]))
                dataList.append(d)

            k = 25

            for idx in range(k - 1, -1, -1):
                current_row = [[d[idx][1], "{0:.3f}".format(float(d[idx][0]))] for d in dataList]
                current_row = list(itertools.chain.from_iterable(current_row))

                result_writer.writerow(current_row)

        print('TopNeg file has been saved.. to', filename_save)

        filename_save = './experiments/results/bias/TopAll_{}.csv'.format(
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
                d.sort(key=lambda x: float(x[0]))
                dataList.append(d)

            # k = 25
            k = len(dataList[0])

            for idx in range(k - 1, -1, -1):
                current_row = [[d[idx][1], "{0:.3f}".format(float(d[idx][0]))] for d in dataList]
                current_row = list(itertools.chain.from_iterable(current_row))

                result_writer.writerow(current_row)

        print('TopNeg file has been saved.. to', filename_save)
