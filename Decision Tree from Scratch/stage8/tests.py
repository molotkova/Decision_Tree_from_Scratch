from hstest import *


ANSWER = ['Made split: Sex is 0',
          'Made split: Age is 0.83',
          'Made split: Fare is 9.0',
          'Made split: Fare is 7.2292',
          'Made split: Age is 28.0',
          'Made split: Age is 22.0',
          'Made split: SibSp is 0',
          'Made split: Age is 34.0',
          'Made split: Age is 21.0',
          'Made split: Pclass is 3',
          'Made split: Fare is 10.5',
          'Made split: Pclass is 2',
          'Made split: Fare is 27.7208',
          'Made split: Fare is 35.5',
          'Made split: Fare is 15.2458',
          'Made split: SibSp is 0',
          'Made split: Fare is 21.075',
          'Made split: Fare is 16.7',
          'Prediction for sample # 0',
          'Considering decision rule on feature Sex with value 0',
          'Considering decision rule on feature SibSp with value 0',
          'Predicted label: 1',
          'Prediction for sample # 1',
          'Considering decision rule on feature Sex with value 0',
          'Considering decision rule on feature Age with value 0.83',
          'Considering decision rule on feature Fare with value 9.0',
          'Considering decision rule on feature Fare with value 7.2292',
          'Predicted label: 0']



class TreeNumTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the files")
        output = pr.execute("test/data_stage8_train.csv test/data_stage8_test.csv").strip()
        res = output.split("\n")
        res = [x.strip() for x in res]
        if len(res) != 28:
            raise WrongAnswer("Wrong number of log messages.\n"
                              "You should make 18 splits building the tree (on given data).\n"
                              "Your model should visit two internal nodes making prediction for the first sample\n"
                              "and four internal nodes making prediction for the second sample.\n"
                              "Do not forget to add log messages showing the number of sample and its predicted label.")
        for i in range(28):
            if i == 18 or i == 22:
                continue
            if res[i] != ANSWER[i]:
                if i == 21:
                    raise WrongAnswer("Wrong prediction for the first sample."
                                      "\nCorrect message template: 'Predicted label: 0'.")
                elif i == 27:
                    raise WrongAnswer("Wrong prediction for the second sample."
                                      "\nCorrect message template: 'Predicted label: 0'.")
                elif i <= 17:
                    raise WrongAnswer(f"Wrong log message on the line {i+1}."
                                      "\nIt may be useful to check the traversal order: go to the left child first."
                                      "\nCorrect message template: 'Made split: Sex is 1'.")
                else:
                    raise WrongAnswer(f"Wrong log message on line {i+1}."
                                      "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first."
                                      "\nCorrect message template: 'Considering decision rule on feature Sex with value 1'.")
        return CheckResult.correct()


if __name__ == '__main__':
    TreeNumTest().run_tests()
