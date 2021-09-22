import LogisticRegression
import NaiveBayes


if __name__ == "__main__":

    while True:
        a = input("Press [1] for Naive Bayes example and [2] for LogisticRegression example or [q] for quit.")
        if a == "1":
            NaiveBayes.run_example()
        elif a == "2":
            LogisticRegression.run_example()
        else:
            print("Exiting")
            break

