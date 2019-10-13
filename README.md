## Kaggle "What's Cooking?" Competition
This was my implementation of a concluded online programming competition
that dealt with cuisine prediction based on lists of ingredients. 
This implementation was done as part of my studies in a Graduate-level program at Rowan University. 

## Usage 
Install needed libs:
```shell script
pip3 install -r requirements.txt
```

The user can use this in a couple of ways. First, they can 
prompt the script with a list of ingredients on the command line 
as a kind of quick test.
```shell script
./contest.py -i water salt "olive oil" vineger oregano
``` 
Note that I used quotes to surround an ingredient that had a space in its text. 

The usage more akin to the original contest is to input a test file 
containing json. This json should have recipe ids and ingredient lists, but not
cuisines, since this program is supposed to guess them. 
```shell script
./contest.py -t /path/to/test/file.json
```

By default, the results will be printed out, but you will likely want an output .csv file.
You can get this by specifying it as another parameter.
```shell script
./contest.py -t /path/to/test/file.json -o /path/to/output.csv
```

I have included a test input and test output file in this repository that correspond with each other. 

### Dependencies
* Python 3
