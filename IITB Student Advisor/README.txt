To run the code-
download all libraries from requirements.txt
make a .env file with contents
GEMINI_API_KEY= "YOUR GEMINI_API_KEY"
The program uses KNN imputation if the optional values are not filled (and also other preprocessing steps)
Then the code calculate the difference between a value (given by user) and the 75th percentile threshold of that value (if the value inputted is lower than threshold)
This difference is multiplied by the weight of the feature to choose which features the LLMS should address in its response
It also calculates the strengths similar to how it calculates these weaknesses, if there are only 0 or 1 weaknesses then it addresses the strengths too
I did this because if it had no strengths given sometimes it would randomly criticise a feature