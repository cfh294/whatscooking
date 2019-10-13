#!/usr/bin/env python3
import pathlib
import csv
import json
import argparse
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

_root = pathlib.PurePath(__file__).parent


class CuisineGuesser(MultinomialNB):
    """
    A big handler to wrap all of our sklearn-related logic
    """
    def __init__(self, in_training, target_key, text_key, id_key="id"):
        """
        Constructor, initialize all values
        :param in_training: the input training dataset
        :param target_key:  the keyname for the target values in the training json
        :param text_key:    the keyname for the text values in the training json
        :param id_key:      the keyname for the id values in the training json
        """
        super().__init__()

        self.__training_data = in_training
        self.__vectorizer = CountVectorizer()
        self.__transformer = TfidfTransformer()

        self.__id_column = id_key
        self.__text_column = text_key
        self.__target_column = target_key
        self.__ids = []
        self.__targets = []
        self.__corpus = []
        self.__preprocess_data(in_training)
        self.train()

    def __tfidf_features(self, in_data):
        """
        Vectorize our data
        :param in_data: the data being vectorized (likely the corpus)
        :return:        the vectorized data
        """
        return self.__vectorizer.fit_transform(in_data).astype("float16")

    def __preprocess_data(self, in_data):
        """
        Append data to our corpus, ids and targets
        :param in_data: the new json data
        :return:
        """
        corpus = []
        for item in in_data:
            self.__ids.append(item[self.__id_column])
            self.__targets.append(item[self.__target_column])
            clean = []
            for old in item[self.__text_column]:
                clean.append(old.replace(" ", "_"))
            corpus.append(" ".join(clean))
        self.__corpus.extend(corpus)

    def train(self):
        """
        Using our corpus, train this model
        :return:
        """
        trained_features = self.__tfidf_features(self.__corpus)
        x_train_tfidf = self.__transformer.fit_transform(trained_features)
        self.fit(x_train_tfidf, self.__targets)

    def add_training_data(self, in_data):
        """
        Add some new training data and then train the model
        :param in_data: the new training data
        :return:
        """
        self.__preprocess_data(in_data)
        self.train()

    def guess_cuisine(self, ingredient_list):
        """
        Guess the cuisine of a recipe based on an incoming list of ingredients
        :param ingredient_list: the ingredient list
        :return: the guess
        """
        clean = []
        for ing in ingredient_list:
            clean.append(ing.replace(" ", "_"))
        clean = [" ".join(clean)]
        new_counts = self.__vectorizer.transform(clean)
        new_tfidf = self.__transformer.transform(new_counts)
        guess = self.predict(new_tfidf)
        return guess[0] if guess else ""


def with_cmd_line_args(f):
    """
    Decorator that passes along parsed command line arguments to the wrapped function
    :param f: a function to be decorated
    :return:  the decorated function
    """
    def with_cmd_line_args_(*args, **kwargs):
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--test_file", help="Path to an input test file.", type=str, required=False)
        ap.add_argument("-o", "--output", help="Output path for output .csv file, if wanted. Prints to command line "
                                               "by default.", type=str, required=False)
        ap.add_argument("-i", "--ingredients", help="List of ingredients", nargs="+", required=False)
        return f(ap.parse_args(), *args, **kwargs)
    return with_cmd_line_args_


def load_json(in_file_path):
    """
    Loads json in from a file located at the given path
    :param in_file_path: a file path
    :return:             loaded json
    """
    with open(in_file_path, "r") as json_file:
        return json.load(json_file)


def process_input_row(in_row, guesser_client):
    """
    Processes a row of data from an input test file
    :param in_row:         the row being processed
    :param guesser_client: an instance of the CuisineGuesser class
    :return:               a dict with row info
    """
    ingredients = in_row["ingredients"]
    row_id = in_row["id"]
    guess = guesser_client.guess_cuisine(ingredients)
    print(f"Recipe ID {row_id}:")
    print(f"\tIngredients: {', '.join(ingredients)}")
    print(f"\tCuisine Guess: {guess}")
    guess_row = dict(
        id=row_id,
        ingredients=", ".join(ingredients),
        guess=guess
    )
    return guess_row


def write_csv_file(out_path, out_rows):
    """
    Writes out the result of this program to a csv file
    :param out_path: the output path for the csv file
    :param out_rows: the data to be written
    :return:
    """
    with open(out_path, "w") as csv_file:
        fields = ["id", "ingredients", "guess"]
        w = csv.DictWriter(csv_file, fieldnames=fields, quotechar="\"")
        w.writeheader()
        w.writerows(out_rows)


@with_cmd_line_args
def main(cmd_line):
    """
    Container function for the main method, contains actual logic of the program
    :param cmd_line: argparse.Namespace object containing our command line arguments
    :return:
    """

    # bring in our training data and create a guesser
    training_file = load_json(str(_root / "train.json"))
    guesser = CuisineGuesser(training_file, "cuisine", "ingredients")

    # user specified a list of ingredients by hand on the command line
    if cmd_line.ingredients:
        print(f"Ingredients: {', '.join(cmd_line.ingredients)}")
        print(f"Cuisine Guess: {guesser.guess_cuisine(cmd_line.ingredients)}")

    # user input a test file with cuisine-less recipes
    if cmd_line.test_file:
        rows = []
        for row in load_json(cmd_line.test_file):
            rows.append(process_input_row(row, guesser))

        # user wants a csv file
        if cmd_line.output:
            write_csv_file(cmd_line.output, rows)


# run the program
if __name__ == "__main__":
    main()

