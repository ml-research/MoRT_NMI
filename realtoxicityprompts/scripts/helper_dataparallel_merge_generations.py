
base_dir = './resultsMCM/prompted/gpt2mcm-k50-keepmin5-t00'
base_path = base_dir+'/{}-{}/generations.jsonl'
outfile = base_dir+'/all/generations.jsonl'
batch_size = 4000
filenames = [(i*batch_size, i*batch_size+batch_size) for i in range(25)]
print(filenames)
cnt = 0

with open(outfile, 'w') as outfile:
    for i, fname_ in enumerate(filenames):
        start = fname_[0]
        end = fname_[1]
        if i + 1 == len(filenames):
            end = 'end'

        fname = base_path.format(start, end)
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
                cnt += 1
print("Finished merging generations: #{}".format(cnt))