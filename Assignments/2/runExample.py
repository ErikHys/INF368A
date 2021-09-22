import LogisticRegression
import NaiveBayes


if __name__ == "__main__":

    while True:
        a = input("Press [1] for Naive Bayes example and [2] for LogisticRegression example or [q] for quit.")
        if a == "1":
            b = input("[1] for Task 1 Assignment 2 example or [2] Lecture 4 slide 46 example")
            NaiveBayes.run_example() if b == "1" else NaiveBayes.run_lecture_example()
        elif a == "2":
            LogisticRegression.run_example()
        else:
            print("Exiting")
            break

