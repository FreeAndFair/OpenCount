import sys

def read_cvr(input_file):
    def process_line(line):
        cvr = line
        entries = cvr.split(',')
        filename = entries[0]
        contests = []
        targets = []
        for i in range(1,len(entries)):
            entry = entries[i].strip() # should remove \t,\n,\r etc...
            if entry == 'over' or entry == 'under' or entry == 'OK':
                contests.append((targets, entry))
                targets = []
            else:
                targets.append(entry)
        return (filename, contests)
    
    cvr = {}
    contest_def = None
    with open(input_file) as f:
        for line in f:
            if line[0] != '#':
                filename, contests = process_line(line)
                cvr[filename] = (contests, line)
    return cvr

def compare_cvrs(true_cvr, test_cvr):
    def prepare_cvr(cvr):
        return
        
    f = open('cvr_compare.txt', 'w')
    
    lines = []
    print >>f, 'missing from test'
    for filename in true_cvr.keys():
        if not test_cvr.has_key(filename):
            contests, line = true_cvr[filename]
            lines.append(line.strip())
    lines.sort()
    for line in lines:
        print >>f, line
    print >>f, ''
    
    lines = []
    print >>f, 'missing from truth'
    for filename in test_cvr.keys():
        if not true_cvr.has_key(filename):
            contests, line = test_cvr[filename]
            lines.append(line.strip())
    lines.sort()
    for line in lines:
        print >>f, line
    print >>f, ''
    
    print >>f, 'differences'
    for filename in true_cvr.keys():
        if test_cvr.has_key(filename):
            true_contests, true_line = true_cvr[filename]
            test_contests, test_line = test_cvr[filename]
            if len(true_contests) != len(test_contests): # different number of contests
                print >>f, 'true: ' + true_line.strip()
                print >>f, 'test: ' + test_line.strip()
                continue
            proceed = True
            for i in range(len(true_contests)):
                if proceed:
                    true_contest = true_contests[i]
                    test_contest = test_contests[i]
                    if len(true_contests) != len(test_contests): # different length contests
                        print >>f, 'true: ' + true_line.strip()
                        print >>f, 'test: ' + test_line.strip()
                        proceed = False
                        continue
                    for j in range(len(true_contest)):
                        if proceed:
                            if true_contest[j] != test_contest[j]: # different results
                                print >>f, 'true: ' + true_line.strip()
                                print >>f, 'test: ' + test_line.strip()
                                proceed = False
                                continue
    return

def create_output(results):
    f = open('cvr_compare.txt', 'w')
    for i in range(len(results)):
        print >>f, text

if __name__ == '__main__':
    script_name, true_filename, test_filename = sys.argv
    true_cvr = read_cvr(true_filename)
    test_cvr = read_cvr(test_filename)
    compare_cvrs(true_cvr, test_cvr)
    
