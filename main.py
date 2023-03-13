import SeqNet

model_list = {'SeqNet': SeqNet}

_, k_r, k_a = SeqNet.train()
print(SeqNet.test('Train'))
print(SeqNet.test('Test'))
SeqNet.draw('Test')
