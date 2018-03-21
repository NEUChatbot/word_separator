file_list = ['msr_training.txt', 'pku_training.txt']

signs = ",.';!，。？！、”“‘’"

def main():
    with open('training_data.txt', 'w') as out:
        for file in file_list:
            with open(file) as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        if word in signs:
                            out.write('\n')
                        elif len(word) == 0:
                            continue
                        elif len(word) == 1:
                            out.write(word + '/S')
                        else:
                            out.write(word[0] + '/B')
                            for c in word[1:-1]:
                                out.write(c + '/M')
                            out.write(word[-1] + '/E')
                    #out.write('\n')

def remove_space():
    with open('characters.txt', 'w') as out:
        for file in file_list:
            with open(file) as f:
                for line in f:
                    words = line.split()
                    out.write(''.join(words))
                    out.write('\n')


def test():
    with open('training_data.txt') as f:
        while True:
            batch = f.read(3)
            if batch is None or batch == '':
                break
            print(batch)

if __name__ == '__main__':
    main()
    # test()
    # remove_space()
    # with open('characters.txt') as f:
    #     for l in f:
    #         print(l)