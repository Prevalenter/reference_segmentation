import yaml

def para_load(path):
	with open(path) as file:
	    fruits_list = yaml.load(file, Loader=yaml.FullLoader)
	    return fruits_list

def para_write(para, path):
	with open(path, 'w') as file:
	    documents = yaml.dump(para, file)

if __name__ == '__main__':
	root = '../data/Drosophila_256_False/para.yaml'
	print(para_load(root))
	para_write(para_load(root), 'test.yaml')
