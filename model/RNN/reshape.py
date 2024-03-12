import os
for path, dir, files in os.walk('./data'):
    print(path, dir, files)
    if len(dir) == 0:
        for file in files:
            if file[-4:] == '.csv':
                # print('csv')
                continue
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f1:
                lines = f1.readlines()[1:]
            newlines = []
            for line in lines:
                newline = line.replace(',', '，').replace('\t', ',').replace('\"', '')
                newlines.append(newline)
            with open(os.path.join(path, file[:-4] + '.csv'), 'w', encoding='utf-8') as f2:
                f2.writelines(newlines)