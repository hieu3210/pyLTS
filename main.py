# Main file of LTS procedure

f = open('datasets/alabama.txt', 'r')

data = list(map(int, f.readline().split(',')))
lb = f.readline()
ub = f.readline()

print(data)
