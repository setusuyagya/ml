import csv
hypo=[]
data=[]
with open('C:/Users/Setu Suyagya/Downloads/Data Sets/sp.csv') as csv_file:
    fd = csv.reader(csv_file)
    print("\nThe given training examples are:")
    for line in fd:
        print(line)
        if line[-1]== "Yes":
            data.append(line)
print("\nThe positive examples are: Enjoy swimming");
for x in data:
    print(x)
row= len(data)
col=len(data[0])
for j in range(col):
    hypo.append(data[0][j])
for i in range(row):
    for j in range(col):
        if hypo[j]!=data[i][j]:
            hypo[j]='?'
print("\nThe maximally specific Find-s hypothesis for the given training examples is")
print(hypo)