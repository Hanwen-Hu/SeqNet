import SeqNet as seq

a, m, s = seq.load_data()
a = seq.generate_set(a)
model = seq.train(a, 1)
print(seq.test(1))

