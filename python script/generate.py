from itertools import islice
import csv
import string
import re
import sentiment
import NLTKAPI


# input files path

NEGATIVE_FILE = '/Users/junhanliu/Desktop/4107/resource/Negative.csv'
POSITIVE_FILE = '/Users/junhanliu/Desktop/4107/resource/Positive.csv'
STOP_WORD_FILE = '/Users/junhanliu/Desktop/4107/resource/StopWords.csv'
WORD_LIST_FILE = '/Users/junhanliu/Desktop/4107/resource/TaboadaGrieve2004-SO.csv'
POLARTITY_FILE = '/Users/junhanliu/Desktop/4107/resource/polarity_list.txt'
TWITTWE_FILE = '/Users/junhanliu/Desktop/4107/resource/semeval_twitter_data.arff.txt'

OUT_PATH = '/Users/junhanliu/Desktop/twitter.arff'


#dictionary
pos_dic = {}
neg_dic = {}
stop_word_dic = {}
word_dic = {}
polarity_dic = {}


def autoLoad():
    print('\nloading positive/nagetive words to dictionary')
    with open(POSITIVE_FILE) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                result = re.sub('[^A-Za-z]+', '', row[0])
                pos_dic[result.lower()] = None
    f.close()
#    for row in pos_dic.keys(): print (row)
    with open(NEGATIVE_FILE) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                result = re.sub('[^A-Za-z]+', '', row[0])
                neg_dic[result.lower()] = None
    f.close()
    with open(STOP_WORD_FILE) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                stop_word_dic[row[0].lower()] = None
    f.close()

    with open(WORD_LIST_FILE) as f:
        reader = csv.reader(f, delimiter=',')
        for row in islice(reader, 2, None):
            if row and row[7]:
                word_dic[row[0].lower()] = float(row[7])
    f.close()

    with open(POLARTITY_FILE) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                list = row[0].split(' ')
                if list[0] and list[1]:
                    polarity_dic[list[0].lower()] = float(list[1])

    f.close()
    print('finished loading')

#    for row in polarity_dic.keys(): print(row+","+str(polarity_dic[row]))


class ARFF:

    def __init__(self, output_file):
        self.arff_file = open(output_file, 'w+')

#        self.arff_file.write('@relation opinion\n@attribute category {positive,negative,neutral,objective}\n@attribute pnumber numeric\n@attribute nnumber numeric\n@attribute weight real\n@attribute polarity real\n@attribute question_mark {yes, no}\n@data\n')
        self.arff_file.write('@relation opinion\n@attribute category {positive,negative,neutral,objective}\n@attribute neg_score real\n@attribute neu_score real\n@attribute pos_score real\n@attribute compund_score real\n@data\n')

    def write_row(self, data):
        self.arff_file.write(data)



class Generate(ARFF):

    def __init__(self, output_file):
        ARFF.__init__(self, output_file)

    def parse(self, data):
        pcounter = 0
        ncounter = 0
    
        wordList = re.sub("[^\w]", " ",  data[0]).split()
        
        #count positive word number
        for item in wordList:
            if item.lower() in pos_dic.keys():
                pcounter += 1
        #count negative word number
        for item in wordList:
            if item.lower() in pos_dic.keys():
                ncounter += 1
        
#        #remove hyperlink
#        string = self.remove_hyperlink(data[0])
#
#        #remove stop words, special characters
#        string2 = self.remove_special_charater(string)

        #calculate weight
        weight = self.cal_weight(data[0])
        
        #calculate another polarity
#        polarity = self.cal_polarity(data[0])
        polarity = sentiment.main(data[0])

        #check if contains emoji
        
        #check if contains ?
        boolean = self.contain_question(data[0])
        #check if contains !
        
        # get string sentiment score
        result = NLTKAPI.get_score(data[0])
        

        self.write_row(data[1]+","+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+'\n')
        


    def contain_question(self, data):
        if '?' in data:
            return 'yes'
        else:
            return 'no'

    def remove_hyperlink(self,data):
        return re.sub(r"http\S+", "", data)

    def remove_special_charater(self, data):
        wordList = re.sub("[^\w]", " ",  data).split()
        for words in wordList:
            if words in stop_word_dic.keys():
                data.replace(words, '')

        result = re.sub('[^ A-Za-z0-9]+', '', data)
        return result

    def cal_weight(self,data):
        weight = 0.0
        wordList = re.sub("[^\w]", " ",  data).split()
    
        for word in wordList:
            if word.lower() in word_dic.keys():
                weight += word_dic[word.lower()]
        return weight

    def cal_polarity(self, data):
        weight = 0.0
        wordList = re.sub("[^\w]", " ",  data).split()
        for word in wordList:
            if word.lower() in polarity_dic.keys():
                weight += polarity_dic[word.lower()]
        return weight






def main():
    autoLoad()

    print('I am extracting featuers, please wait...')
    with open(TWITTWE_FILE) as f:
        reader = csv.reader(f, delimiter=',', quotechar='\'')

        generate = Generate(OUT_PATH)

        for record in islice(reader, 4, None):
            generate.parse(record)

        f.close()




main()















