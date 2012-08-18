from PIL import Image, ImageDraw
import os
from random import random
from collections import Counter
import multiprocessing as mp

black = 200

do_save = False
export = True

def num2pil(img):
    pilimg = Image.new("L", (len(img[0]), len(img)))
    pilimg.putdata([item for sublist in img for item in sublist])
    return pilimg

def load_pil(path):
    pilimg = Image.open(path)
    pilimg = pilimg.convert("L")
    return pilimg

def load_num(path="", pilimg=None):
    if pilimg == None:
        pilimg = load_pil(path)
    width, height = pilimg.size
    data = list(pilimg.getdata())
    data = [data[x:x+width] for x in range(0,width*height,width)]
    #print width, height
    return data

def load_threshold(image):
    dat = load_num(image)
    dat = [[x < black for x in y] for y in dat]
    block = 10
    boxes = {}
    for y in range(0,len(dat)-block, block):
        for x in range(0,len(dat[y])-block, block):
            filled = sum(dat[y+dy][x+dx] for dy in range(block) for dx in range(block)) > block*block*9/10
            if filled:
                boxes[x,y] = True
    remove = []
    for x,y in boxes:
        if (x,y-block) in boxes and (x,y+block) in boxes:
            if (x-block,y) in boxes and (x+block,y) in boxes:
                remove.append((x,y))
    for x,y in remove:
        for dy in range(block):
            for dx in range(block):
                dat[y+dy][x+dx] = False

    
    dat = [[0 if x else 255 for x in y] for y in dat]
    if do_save:
        load_pil(image).save("tmp/%s-a.png"%image.split("/")[1])
        num2pil(dat).save("tmp/%s-b.png"%image.split("/")[1])
    return dat

def find_lines(data):
    """
    Find all the lines we can on the image.

    For each pixel, if it is black:
        (1) Extend up and down as far as possible. Call those
              pixels part of one line.
        (2) Extend left and right as far as possible. Call those
              pixels part of one line.
        (3) Mark the lined pixels as 'used'.
    
    When extending, first move to the middle of the line. Then,
    do a DFS search going in the direction of the line, but only
    in the given direction.

    In order to improve efficiency, only test the black pixels on a grid.
    """
    height, width = len(data), len(data[0])
    def extend_ud(point):
        y, x = point
        while 0 < y and data[y][x] < black: y -= 1
        upper_y = y
        y, x = point
        low = len(data)
        while y < low and data[y][x] < black: y += 1
        lower_y = y
        return upper_y, lower_y
    def extend_lr(point):
        y, x = point
        while 0 < x and data[y][x] < black: x -= 1
        left_x = x
        y, x = point
        right = len(data[y])
        while x < right and data[y][x] < black: x += 1
        right_x = x
        return left_x, right_x

    def full_extend_ud(point):
        y, x = point
        l,r = extend_lr((y,x))
        x = (l+r)/2 if l+r<20 else x
        u1,d1 = extend_ud((y,x))
        return u1,d1

    def full_extend_lr(point):
        y, x = point
        u,d = extend_ud((y,x))
        y = (u+d)/2 if u+d<20 else y
        l1,r1 = extend_lr((y,x))
        return l1,r1

    LST = []
    def full_extend_lr_2(point):
        u,d = extend_ud(point)
        if d-u < 20: y = (u+d)/2
        else: y = point[0]
        point = (y,point[1])

        q = [point[0]]
        x = point[1]-1
        while q and x > 0:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if 0 <= dy+y < height and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x -= 1
        l = x
        q = [point[0]]
        x = point[1]+1
        while q and x < width:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if 0 <= dy+y < height and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x += 1
        r = x

        return l,r

    def full_extend_ud_2(point):
        l,r = extend_lr(point)
        if r-l < 20: x = (l+r)/2
        else: x = point[1]
        point = (point[0],x)

        q = [point[1]]
        y = point[0]-1
        while q and y > 0:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if 0 <= dx+x < width and data[y][dx+x] < black]))
            #LST.extend([(x,y) for y in q])
            y -= 1
        u = y
        q = [point[1]]
        y = point[0]+1
        while q and y < height:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if 0 <= dx+x < width and data[y][dx+x] < black]))
            #LST.extend([(x,y) for y in q])
            y += 1
        d = y

        return u,d

    foundy = {}
    foundx = {}
    lines = []
    YSKIP = 15
    XSKIP = 40
    for y in range(0,height,1):
        for x in range(0,width,1) if y%YSKIP == 0 else range(0,width,XSKIP):
            if not data[y][x] < black: 
                #data[y][x] = black
                continue
            if y%YSKIP == 0 and (y/3,x/3) not in foundy:
                u,d = full_extend_ud((y,x))
                if d-u > 30:
                    u,d = full_extend_ud_2((y,x))
                    if d-u > 30:
                        for dx in range(-10, 10, 3):
                            for q in range(u,d):
                                foundy[q/3,(x+dx)/3] = True
                        lines.append(("V", (x-3,u,x+3,d)))

            if x%XSKIP == 0 and (y/3,x/3) not in foundx:
                #print 'h', newy, y, x
                l,r = full_extend_lr_2((y,x))
                if r-l > 30:
                    for dy in range(-10, 10, 3):
                        for q in range(l,r):
                            foundx[(y+dy)/3,q/3] = True
                    #print 'line starting from', x, y, data[y][x]
                    #LST.append((x-3,y-3,x+3,y+3))
                    #LST.append((l,y,r,y))
                    lines.append(("H", (l,y-3,r,y+3)))
    
    if do_save:
        num2pil(data).save(tmp+"/it.png")
        ct = Counter(LST)
        im = Image.new("RGB", (width, height), (255,255,255))
        d = ImageDraw.Draw(im)
        LST = list(set(LST))
        for each in LST:
            if len(each) == 4:
                d.rectangle(each, fill=(0,0,0))
            else:
                d.point(each, fill=(ct[each],0,0))                
        im.save(tmp+"/asdf.png")

        im = Image.new("L", (width, height), 255)
        d = ImageDraw.Draw(im)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] < black:
                    d.point((j,i), fill=0)
        im.save(tmp+"/asdf2.png")
    

    #print len(lines)
    #print lines
    return lines


def intersect(line1, line2):
    """
    Compute the intersection of two bounding boxes.
    """
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None
def union(line1, line2):
    """
    Compute the union of two bounding boxes.
    """
    top = min(line1[1], line2[1])
    bottom = max(line1[3], line2[3])
    left = min(line1[0], line2[0])
    right = max(line1[2], line2[2])
    return left, top, right, bottom

def dfs(graph, start):
    """
    Run a DFS search on a graph starting at a given vertex.
    Return the list of seen graphs.
    """
    s = [start]
    seen = {}
    while s != []:
        top = s.pop()
        if top in seen: continue
        seen[top] = True
        s += graph[top]
    return seen.keys()

def to_graph(lines, width, height):
    """
    Convert a set of lines to graph where lines are vertexes, and
    there is an edge between two lines when they intersect. This
    prepares for finding all rectangles.

    First, we need to find lines which were accidentally split in to
    pieces, and merge them together. That is, if there are two
    horizontal lines which intersect, or two vertical lines which
    intersect, then merge them in to one line of the union of the two.
    
    We do this by making another graph of all horizontal lines and 
    adding edges between the lines when they touch. Then we run a
    connnected components algorithm over it and take the union.
    This is done by creating an width*height array, storing where all
    of the lines are, and finding when they touch.

    Then it is easy to do an all-pairs algorithm to find the intersecting
    vertical and horizontal lines.
    """
    print width, height

    # Extend the lines.
    def extend(line):
        if line[0] == 'V':
            l, u, r, d = line[1]
            ext = int(round((d-u)*0.02))
            return ('V', (l, u-ext if u-ext >= 0 else 0, r, d+ext if d+ext < height else height-1))
        if line[0] == 'H':
            l, u, r, d = line[1]
            ext = int(round((r-l)*0.02))
            return ('H', (l-ext if l-ext >= 0 else 0, u, r+ext if r+ext < width else width-1, d))

    for _ in range(2):
        print "THERE ARE", len(lines)
        lines = map(extend, lines)
    
        for direction in ['H', 'V']:
            table = [[None]*(width+20) for _ in range(height+20)]
            equal = []
            for full in lines:
                if full[0] != direction: continue
                _,(l,u,r,d) = full
                for x in range(l,r):
                    for y in range(u,d):
                        if table[y][x] != None:
                            equal.append((table[y][x], full))
                        else:
                            table[y][x] = full
            equal = list(set(equal))
            #print equal
            graph = {}
            for v1,v2 in equal:
                if v1 not in graph: graph[v1] = []
                if v2 not in graph: graph[v2] = []
                graph[v1].append(v2)
                graph[v2].append(v1)
            #print graph
            seen = {}
            new = []
            for el in graph.keys():
                if el in seen: continue
                makeequal = dfs(graph, el)
                for each in makeequal:
                    seen[each] = True
                new.append((makeequal[0][0], reduce(union, [x[1] for x in makeequal])))
            for line in lines:
                if line not in seen:
                    new.append(line)
            lines = new
    print "THERE ARE END", len(lines)
        
    vertexes = dict((x, []) for _,x in lines)

    boxes = []
    for way1,line1 in lines:
        for way2,line2 in lines:
            if way1 != way2:
                if intersect(line1, line2):
                    boxes.append(intersect(line1, line2))
                    vertexes[line1].append(line2)
    print 'finished'
    return boxes,dict((k,v) for k,v in vertexes.items() if v != [])

def find_squares(graph):
    """
    Given a graph (vertexes are lines, edges when they intersect),
    return the squares that are in the graph.
    A square is when the DFS finds a back-edge where the difference in
    the preorders of the two nodes is 4.
    """
    def dfs_square(stack, debug=False):
        if debug:
            print ".  "*len(stack), stack[-1]
        if len(stack) == 4:
            if stack[0] in graph[stack[-1]]:
                return [intersect(union(stack[0],stack[2]), union(stack[1],stack[3]))]
            return [None]
        res = []
        for vertex in graph[stack[-1]]:
            if vertex in stack: continue
            res += dfs_square(stack+[vertex], debug)
        return res

    return list(set([x for x in sum([dfs_square([start]) for start in graph], []) if x]))

def do_extract(name, img, squares, giventargets):
    """
    Find all contests and extract them.
    
    Start with the smallest sized bounding box, and check if it
    contains any voting targets. 

    If it does, it's a contest. Remove it and the targets that
    are inside of it. Move on to next biggest.
    
    If not, then remove it and move on to the next biggest.

    For each contest, see if there are any previously-identified
    voting targets that were removed by a smaller bounding box,
    if there are, warn the operator. This means one contest
    encloses another.
    """
    def area(x): 
        if x == None: return 0
        return (x[2]-x[0])*(x[3]-x[1])

    targets = [x for x in giventargets]
    avg_targ_area = sum(map(area,targets))/len(targets)
    squares = [x for x in squares if area(x) > avg_targ_area*2]

    contests = []

    #print "T", targets
    for sq in sorted(squares, key=area):
        if sq in targets: continue
        inside = [t for t in targets if area(intersect(sq, t)) == area(t)]
        if inside != []:
            contests.append(sq)
            targets = [x for x in targets if x not in inside]

    keepgoing = True
    while keepgoing:
        keepgoing = False
        for target in giventargets:
            tomerge = [x for x in contests if intersect(x, target)]
            if len(tomerge) != 1:
                print "MERGING", tomerge
                contests = [x for x in contests if x not in tomerge] + [reduce(union, tomerge)]
                keepgoing = True
                break
    
    #print "C", contests
    for cont in contests:
        if export:
            im = img.crop(cont)
            name = tmp+"/"+str(sum(im.histogram()[:100]))+".png"
            im = img.crop(cont)
            im.save(name)

    if do_save:
        new = Image.new("RGB", img.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for box in contests:
            imd.rectangle(box, outline=(0,0,255))
        print "GIVEN", giventargets
        for box in giventargets:
            imd.rectangle(box, fill=(255,0,0))
        new.save(tmp+"/qqq.png")

    return contests

    #print targets, contests
    #os.popen("open tmp/*")
    #exit(0)
        

def extract_contest(args):
    if len(args) == 2:
        image_path, giventargets = args
        returnimage = True
    elif len(args) == 3:
        image_path, giventargets, returnimage = args
    else:
        raise Error("Wrong number of args")
    print "processing", image_path
    #Image.open(image_path).save(tmp+"/"+image_path.split("/")[-1][:-4]+"-orig.png")
    data = load_threshold(image_path)
    lines = find_lines(data)
    boxes, graph = to_graph(lines, len(data[0]), len(data))
    squares = find_squares(graph)
    squares = sorted(squares, key=lambda x: -(x[2]-x[0])*(x[3]-x[1]))
    #print lines
    #print squares

    if do_save:
        show = num2pil(data)
        new = load_pil(image_path).copy().convert("RGB")#Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in [x[1] for x in lines]:
            imd.rectangle(line, outline=(0,0,0))
        for line in boxes:
            imd.rectangle(line, fill=(0,0,255))
    
        print len(squares), "NUM"
    
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-line.png")
    
        new = Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in graph:
            imd.rectangle(line, outline=(0,0,0))
        for line in boxes:
            imd.rectangle(line, fill=(0,0,255))
    
        print len(squares), "NUM"
    
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-line-2.png")
    
        for l,u,r,d in squares:
            c = (int(random()*255), int(random()*255), int(random()*255))
            imd.rectangle((l,u,r,d), fill=c)
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-box.png")

    if do_save or export:
        loadedimage = load_pil(image_path)
    else:
        loadedimage = None
    final = do_extract(image_path.split("/")[-1], 
                       loadedimage, squares, giventargets)
    if returnimage:
        return data, final
    else:
        return final

def ballot_preprocess(i, f, image, contests, targets, lang):
    """
    Preprocess a ballot and turn it in to its corresponding data.
    For each contest, record the ballot ID, the contest bounding box,
    as well as the text associated with the contest.
    """
    sub = os.path.join(tmp+"", f.split("/")[-1].split(".")[0]+"-dir")
    #print "SUB IS", sub
    os.mkdir(sub)
    res = []
    #print "CONTESTS", contests
    for c in contests:
        #print "TOMAKE", c
        os.mkdir(os.path.join(sub, "-".join(map(str,c))))
        t = compare_preprocess(lang, os.path.join(sub, "-".join(map(str,c))), 
                               image, c, targets)
        res.append((i, c, t))
    #print "RESULTING", res
    return res


def compare_preprocess(lang, path, image, contest, targets):
    """
    Identifies the text associated with the contest.

    Split the contest in to "stripes", one for each voting target,
    and one for the title. OCR the text and record it.
    """
    #print contest, targets
    #print [intersect(contest, x) for x in targets]
    #print 'all', targets
    targets = [x for x in targets if intersect(contest, x) == x]
    l,u,r,d = contest
    cont_area = load_num(pilimg=num2pil(image).crop((l+10,u+10,r-10,d-10)))
    #print "TEXT FOR", contest
    #print "bottom of box", d
    #print 'targets', targets
    tops = [0]+sorted([a[1]-u-10 for a in targets])+[d]
    #print contest
    #print "USING", tops
    blocks = []
    for upper,lower in zip(tops, tops[1:]):
        istarget = (upper != 0)
        if upper == lower:
            blocks.append((istarget, ""))
            continue
        #print "POS", upper, lower
        #print len(cont_area[upper:lower])
        img = num2pil(cont_area[upper:lower])
        name = os.path.join(path, str(upper)+".tif")
        img.save(name)
        os.popen("tesseract %s %s -l %s"%(name, name, lang))
        if os.path.exists(name+".txt"):
            #print "THIS BLOCK GOT", open(name+".txt").read().decode('utf8')
            blocks.append((istarget, open(name+".txt").read().decode('utf8')))
        else:
            print "-"*40
            print "OCR FAILED"
            print "-"*40
            blocks.append((istarget, ""))
            
    
    print blocks
    return blocks

memo = {}
def row_dist(a, b):
    """
    Compute the edit distance between two strings.
    """
    global memo
    if a == b: return 0
    if (a,b) in memo: 
        return memo[a,b]
    table = [[0]*(len(b)+2) for _ in range(len(a)+2)]
    for i in range(len(a)+2): table[i][0] = i
    for i in range(len(b)+2): table[0][i] = i
 
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i][j] = min(table[i-1][j] + 1,
                              table[i][j-1] + 1,
                              table[i-1][j-1] + (a[i-1] != b[j-1]))
    memo[a,b] = memo[b,a] = table[-2][-2]
    return table[-2][-2]

count = 0
def compare(otexts1, otexts2):
    """
    Compute the distance between two contests.
    
    This distance allows the order of the targets to shuffle,
    but does not allow the text within a target to reorder.
    
    We do this with an approximation of the weighted disjoint
    set cover problem. We sort the edges by weight, and take the
    minimum weight one, remove the corresponding nodes, and recurse.
    
    The distance is sum of the edges picked, as well as the sum
    of the unused vertexes.
    """
    #print 'running with1', otexts1
    #print 'running with2', otexts2
    # text -> length
    texts1 = dict((y, len(y)) for x,y in otexts1)
    texts2 = dict((y, len(y)) for x,y in otexts2)
    # text -> target true/false
    istargs1 = dict((y,x) for x,y in otexts1)
    istargs2 = dict((y,x) for x,y in otexts2)
    # Text associated with targets only
    targtext1 = [y for x,y in otexts1 if x]
    targtext2 = [y for x,y in otexts2 if x]
    size = sum(texts1.values())+sum(texts2.values())
    if size == 0:
        print "Possible Error: A contest has no text associated with it"
        return 0, []
    weights = sorted([(row_dist(a,b),a,b) for a in texts1 for b in texts2])

    #for each in enumerate(targtext1): print each
    #for each in enumerate(targtext2): print each

    val = 0
    matching = []
    while texts1 != {} and texts2 != {}:
        for weight,a,b in weights:
            if a in texts1 and b in texts2 and istargs1[a] == istargs2[b]:
                if istargs1[a] == True:
                    #print 'together', targtext1.index(a), targtext2.index(b)
                    matching.append((targtext1.index(a),
                                     targtext2.index(b)))
                val += weight
                del texts1[a]
                del texts2[b]
                break
    #print "MATCHING", matching
    return float(val+sum(texts1.values())+sum(texts1.values()))/size, matching

def first_pass(contests):
    """
    Split a set of contests in to a set of sets, where each
    set contains the same number of voting targets.
    """
    ht = {}
    for each in contests:
        if len(each[2]) not in ht: ht[len(each[2])] = []
        ht[len(each[2])].append(each)
    return ht.values()

def split_to_equal(contests):
    """
    Split a set of contests in to a set of those which are 
    truly equal. Create a set of known equiv-classes, and for
    each target, compare with every class. If it's not in any,
    put it in a class of its own.
    """
    sets = []
    print len(contests)
    print "CONTS", contests
    for i,each in enumerate(contests):
        print 'on', i, '#', len(sets)
        found = False
        for s in sets:
            # get a representitive, then get the non-matching part, then the text
            score, matching = compare(s[0][0][2], each[2])
            #score = compare(s[0][2], each[2])
            if score < .2:
                s.append((each, matching))
                #s.append(each)
                found = True
                break
        if not found:
            sets.append([(each, list(zip(range(len(each[2])), range(len(each[2])))))])
            #sets.append([each])
    return sets
    for s in sets:
        print '='*80
        print 'NEW EQU CLASS'
        print '='*80
        for ss in s:
            print "-"*40
            print ss[0]+"."+str(ss[1])
            
def equ_class(contests):
    #print map(len, contests)
    #print contests
    contests = sum(contests, [])
    #print contests
    groups = first_pass(contests)
    # Each group is known to be different.
    result = []
    for group in groups:
        result += split_to_equal(group)
    print "RETURNING", result
    return result

def merge_contests(ballot_data, fulltargets):
    """
    Given a set of bounding boxes, merge together those which
    are different boundingboxes but are, in reality, the same contest.
    """
    new_data = []
    for ballot, targets in zip(ballot_data, fulltargets):
        #print 'next'
        new_ballot = []
        for group in targets:
            #print 'targs is', group
            equal = [i for t in group for i,(_,bounding,_) in enumerate(ballot) if intersect(t, bounding)]
            equal_uniq = []
            for e in equal:
                if e not in equal_uniq: equal_uniq.append(e)
            #print 'add', equal_uniq
            merged = sum([ballot[x][2] for x in equal_uniq],[])
            new_ballot.append((ballot[equal[0]][0], [ballot[x][1] for x in list(set(equal))], merged))
        new_data.append(new_ballot)
    print new_data
    return new_data

def extend_multibox(ballots, box1, box2, orders):
    ballot = ballots[box1[0]]
    txt1 = [x for x in ballot if x[:2] == box1][0]
    txt2 = [x for x in ballot if x[:2] == box2][0]
    txt = txt1[2]+txt2[2]
    res = []
    for bid,order in enumerate(orders):
        #print 'BID IS', bid
        for c1,c2 in order:
            t1 = [x for x in ballots[bid] if x[:2] == c1][0]
            t2 = [x for x in ballots[bid] if x[:2] == c2][0]
            if len(t1[2])+len(t2[2]) != len(txt1[2])+len(txt2[2]):
                continue
            #print '-'*30
            #print 'consec', c1, c2
            score, _ = compare(txt, t1[2]+t2[2])
            if score < .2:
                #print "THEY ARE EQUAL"
                res.append((c1, c2))
    print "RESULT", res

    return res

def do_grouping(t, paths, giventargets, lang_map = {}):
    global tmp
    print "ARGUMENTS", (t, paths, giventargets, lang_map)
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    ballots = []
    for i,f in enumerate(paths):
        print f
        im, contests = extract_contest((f, sum(giventargets[i],[])))
        lang = lang_map[f] if f in lang_map else 'eng'
        get = ballot_preprocess(i, f, im, contests, sum(giventargets[i],[]), lang)
        ballots.append(get)
    print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets)

def find_contests(t, paths, giventargets):
    global tmp
    #print "ARGS", (t, paths, giventargets)
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    args = [(f, sum(giventargets[i],[]), False) for i,f in enumerate(paths)]
    pool = mp.Pool(mp.cpu_count())
    ballots = pool.map(extract_contest, args)
    print "RETURNING", ballots
    return ballots

def group_given_contests_map(arg):
    i,(f,conts) = arg
    print f
    im = load_num(f)
    lang = lang_map[f] if f in lang_map else 'eng'
    return ballot_preprocess(i, f, im, conts, sum(giventargets[i],[]), lang)
        
def group_given_contests(t, paths, giventargets, contests, lang_map = {}):
    global tmp
    print "ARGUMENTS", (t, paths, giventargets, lang_map)
    #print 'giventargets', giventargets
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    pool = mp.Pool(mp.cpu_count())
    ballots = pool.map(group_given_contests_map, list(enumerate(zip(paths,contests))))
    #print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets)

def final_grouping(ballots, giventargets):
    print "RUNNING FINAL GROUPING"
    ballots = merge_contests(ballots, giventargets)
    print "NOW EQU CLASSES"
    return equ_class(ballots)

#marin/vbm-34.png
#marin/vbm-85.png

#load_threshold('marin/poll-06.png')
#map(load_threshold, ['marin/poll-77.png', 'marin/poll-31.png', 'marin/poll-06.png', 'marin/vbm-60.png', 'marin/poll-87.png', 'marin/poll-55.png', 'marin/vbm-22.png', 'marin/vbm-90.png', 'marin/mail-65.png', 'marin/poll-19.png', 'marin/poll-97.png', 'marin/poll-66.png', 'marin/mail-23.png', 'marin/mail-73.png', 'marin/vbm-96.png', 'marin/vbm-10.png', 'marin/vbm-85.png', 'marin/poll-42.png', 'marin/vbm-34.png', 'marin/vbm-74.png', 'marin/mail-00.png', 'marin/mail-53.png', 'marin/vbm-46.png'])

find_contests('tmp', ['marin/poll-77.png', 'marin/poll-31.png', 'marin/poll-06.png', 'marin/vbm-60.png', 'marin/poll-87.png', 'marin/poll-55.png', 'marin/vbm-22.png', 'marin/vbm-90.png', 'marin/mail-65.png', 'marin/poll-19.png', 'marin/poll-97.png', 'marin/poll-66.png', 'marin/mail-23.png', 'marin/mail-73.png', 'marin/vbm-96.png', 'marin/vbm-10.png', 'marin/vbm-85.png', 'marin/poll-42.png', 'marin/vbm-34.png', 'marin/vbm-74.png', 'marin/mail-00.png', 'marin/mail-53.png', 'marin/vbm-46.png'], [[[(77, 1494, 110, 1515), (76, 1531, 109, 1552), (76, 1569, 109, 1590)], [(75, 1719, 108, 1740), (74, 1757, 107, 1778), (74, 1794, 107, 1815), (73, 1832, 106, 1853), (73, 1870, 106, 1891)], [(84, 855, 117, 876), (83, 893, 116, 914), (82, 931, 115, 952), (83, 968, 116, 989), (82, 1006, 115, 1027), (82, 1043, 115, 1064), (81, 1081, 114, 1102), (81, 1118, 114, 1139)], [(79, 1269, 112, 1290), (79, 1307, 112, 1328), (78, 1344, 111, 1365)], [(798, 713, 831, 734), (1024, 715, 1057, 736)], [(89, 255, 122, 276), (89, 293, 122, 314), (89, 331, 122, 352), (88, 368, 121, 389), (88, 406, 121, 427), (88, 443, 121, 464), (87, 481, 120, 502), (87, 518, 120, 539), (86, 556, 119, 577), (86, 593, 119, 614), (86, 631, 119, 652), (86, 668, 119, 689), (85, 706, 118, 727)], [(801, 376, 834, 397), (1027, 378, 1060, 399)]], [[(81, 861, 114, 882), (81, 899, 114, 920), (82, 936, 115, 957), (81, 974, 114, 995), (82, 1011, 115, 1032), (82, 1049, 115, 1070), (82, 1086, 115, 1107), (82, 1124, 115, 1145)], [(78, 261, 111, 282), (78, 299, 111, 320), (78, 336, 111, 357), (78, 374, 111, 395), (79, 411, 112, 432), (79, 449, 112, 470), (79, 487, 112, 508), (79, 524, 112, 545), (79, 562, 112, 583), (80, 599, 113, 620), (80, 636, 113, 657), (80, 674, 113, 695), (80, 711, 113, 732)], [(84, 1499, 117, 1520), (85, 1536, 118, 1557), (85, 1574, 118, 1595), (85, 1612, 118, 1633)], [(83, 1274, 116, 1295), (83, 1312, 116, 1333), (83, 1349, 116, 1370)], [(1019, 706, 1052, 727), (793, 707, 826, 728)], [(1017, 369, 1050, 390), (792, 370, 825, 391)]], [[(677, 898, 710, 919), (677, 936, 710, 957), (677, 974, 710, 995), (677, 1011, 710, 1032), (678, 1048, 711, 1069), (677, 1086, 710, 1107), (677, 1123, 710, 1144), (678, 1160, 711, 1181), (677, 1198, 710, 1219), (678, 1235, 711, 1256), (677, 1273, 710, 1294), (678, 1310, 711, 1331), (678, 1348, 711, 1369), (677, 1386, 710, 1407), (678, 1424, 711, 1445), (678, 1461, 711, 1482), (678, 1499, 711, 1520), (678, 1536, 711, 1557), (678, 1574, 711, 1595), (678, 1611, 711, 1632), (678, 1649, 711, 1670), (678, 1686, 711, 1707), (678, 1724, 711, 1745), (678, 1762, 711, 1783), (678, 1799, 711, 1820)], [(73, 825, 106, 846), (73, 862, 106, 883), (74, 899, 107, 920), (74, 937, 107, 958)]], [[(676, 896, 709, 917), (676, 933, 709, 954), (676, 971, 709, 992), (676, 1008, 709, 1029), (676, 1046, 709, 1067), (676, 1083, 709, 1104), (676, 1121, 709, 1142), (676, 1158, 709, 1179), (677, 1196, 710, 1217), (677, 1233, 710, 1254), (676, 1271, 709, 1292), (677, 1308, 710, 1329), (676, 1346, 709, 1367), (677, 1383, 710, 1404), (677, 1420, 710, 1441), (677, 1457, 710, 1478), (677, 1495, 710, 1516), (677, 1532, 710, 1553), (677, 1570, 710, 1591), (677, 1608, 710, 1629), (677, 1645, 710, 1666), (677, 1683, 710, 1704), (677, 1720, 710, 1741), (677, 1758, 710, 1779), (677, 1795, 710, 1816)], [(74, 822, 107, 843), (73, 860, 106, 881), (73, 897, 106, 918), (73, 934, 106, 955), (73, 972, 106, 993), (73, 1009, 106, 1030), (73, 1047, 106, 1068)]], [[(82, 864, 115, 885), (82, 902, 115, 923), (82, 939, 115, 960), (83, 976, 116, 997), (83, 1014, 116, 1035), (84, 1051, 117, 1072), (84, 1089, 117, 1110), (85, 1127, 118, 1148)], [(75, 263, 108, 284), (75, 301, 108, 322), (76, 339, 109, 360), (76, 376, 109, 397), (76, 414, 109, 435), (77, 452, 110, 473), (78, 489, 111, 510), (78, 527, 111, 548), (78, 564, 111, 585), (79, 601, 112, 622), (80, 638, 113, 659), (80, 676, 113, 697), (80, 713, 113, 734)], [(89, 1502, 122, 1523), (89, 1540, 122, 1561), (90, 1577, 123, 1598)], [(86, 1277, 119, 1298), (87, 1314, 120, 1335), (87, 1352, 120, 1373)], [(1019, 702, 1052, 723), (793, 705, 826, 726)], [(1015, 365, 1048, 386), (789, 368, 822, 389)], [(1022, 1041, 1055, 1062), (797, 1043, 830, 1064)]], [[(79, 1493, 112, 1514), (79, 1530, 112, 1551), (78, 1568, 111, 1589), (78, 1606, 111, 1627)], [(76, 1831, 109, 1852), (76, 1868, 109, 1889), (75, 1906, 108, 1927)], [(84, 855, 117, 876), (83, 893, 116, 914), (83, 930, 116, 951), (83, 967, 116, 988), (82, 1005, 115, 1026), (82, 1043, 115, 1064), (82, 1080, 115, 1101), (82, 1118, 115, 1139)], [(80, 1268, 113, 1289), (80, 1305, 113, 1326), (80, 1342, 113, 1363)], [(690, 223, 723, 244), (690, 260, 723, 281), (690, 298, 723, 319), (690, 335, 723, 356), (689, 373, 722, 394)], [(89, 255, 122, 276), (88, 293, 121, 314), (89, 330, 122, 351), (88, 368, 121, 389), (88, 406, 121, 427), (88, 443, 121, 464), (87, 481, 120, 502), (87, 518, 120, 539), (87, 555, 120, 576), (86, 593, 119, 614), (86, 630, 119, 651), (86, 667, 119, 688), (85, 705, 118, 726)], [(796, 1011, 829, 1032), (1022, 1012, 1055, 1033)], [(793, 1348, 826, 1369), (1019, 1350, 1052, 1371)], [(799, 673, 832, 694), (1025, 675, 1058, 696)]], [[(678, 896, 711, 917), (678, 933, 711, 954), (678, 971, 711, 992), (678, 1008, 711, 1029), (678, 1045, 711, 1066), (678, 1083, 711, 1104), (678, 1120, 711, 1141), (678, 1158, 711, 1179), (678, 1195, 711, 1216), (678, 1233, 711, 1254), (678, 1271, 711, 1292), (678, 1308, 711, 1329), (678, 1346, 711, 1367), (678, 1382, 711, 1403), (678, 1420, 711, 1441), (678, 1457, 711, 1478), (678, 1495, 711, 1516), (678, 1532, 711, 1553), (678, 1570, 711, 1591), (678, 1607, 711, 1628), (678, 1645, 711, 1666), (678, 1682, 711, 1703), (678, 1720, 711, 1741), (678, 1758, 711, 1779), (678, 1795, 711, 1816)], [(75, 822, 108, 843), (75, 859, 108, 880), (75, 896, 108, 917), (75, 933, 108, 954)]], [[(677, 895, 710, 916), (677, 932, 710, 953), (676, 970, 709, 991), (677, 1007, 710, 1028), (676, 1045, 709, 1066), (677, 1082, 710, 1103), (676, 1120, 709, 1141), (677, 1158, 710, 1179), (677, 1195, 710, 1216), (676, 1233, 709, 1254), (677, 1270, 710, 1291), (677, 1308, 710, 1329), (677, 1345, 710, 1366), (676, 1382, 709, 1403), (677, 1420, 710, 1441), (677, 1457, 710, 1478), (677, 1494, 710, 1515), (676, 1532, 709, 1553), (677, 1569, 710, 1590), (677, 1607, 710, 1628), (677, 1644, 710, 1665), (677, 1682, 710, 1703), (677, 1719, 710, 1740), (677, 1757, 710, 1778), (677, 1794, 710, 1815)], [(74, 821, 107, 842), (74, 859, 107, 880), (74, 896, 107, 917), (74, 933, 107, 954)]], [[(84, 260, 117, 281), (84, 297, 117, 318), (84, 335, 117, 356), (84, 372, 117, 393), (84, 410, 117, 431), (84, 447, 117, 468), (84, 485, 117, 506), (84, 522, 117, 543), (84, 560, 117, 581), (84, 597, 117, 618), (84, 634, 117, 655), (84, 671, 117, 692), (84, 709, 117, 730)], [(84, 859, 117, 880), (85, 896, 118, 917), (84, 934, 117, 955), (84, 971, 117, 992), (85, 1008, 118, 1029), (84, 1046, 117, 1067), (84, 1083, 117, 1104), (84, 1121, 117, 1142)], [(85, 1495, 118, 1516), (85, 1533, 118, 1554), (85, 1571, 118, 1592)], [(85, 1271, 118, 1292), (85, 1309, 118, 1330), (85, 1346, 118, 1367)], [(797, 708, 830, 729), (1023, 708, 1056, 729)], [(797, 372, 830, 393), (1023, 372, 1056, 393)]], [[(81, 1493, 114, 1514), (80, 1531, 113, 1552), (80, 1569, 113, 1590), (80, 1606, 113, 1627)], [(78, 1756, 111, 1777), (78, 1793, 111, 1814), (77, 1831, 110, 1852), (77, 1869, 110, 1890), (77, 1906, 110, 1927)], [(87, 857, 120, 878), (87, 894, 120, 915), (86, 932, 119, 953), (86, 969, 119, 990), (86, 1006, 119, 1027), (85, 1043, 118, 1064), (85, 1081, 118, 1102), (84, 1118, 117, 1139)], [(83, 1268, 116, 1289), (83, 1306, 116, 1327), (82, 1343, 115, 1364)], [(801, 713, 834, 734), (1027, 715, 1060, 736)], [(92, 258, 125, 279), (92, 295, 125, 316), (91, 333, 124, 354), (91, 370, 124, 391), (91, 408, 124, 429), (90, 445, 123, 466), (91, 482, 124, 503), (90, 519, 123, 540), (90, 557, 123, 578), (89, 594, 122, 615), (89, 632, 122, 653), (89, 669, 122, 690), (88, 707, 121, 728)], [(805, 376, 838, 397), (1030, 378, 1063, 399)]], [[(82, 865, 115, 886), (83, 903, 116, 924), (83, 940, 116, 961), (83, 978, 116, 999), (84, 1015, 117, 1036), (84, 1053, 117, 1074), (85, 1091, 118, 1112), (85, 1129, 118, 1150)], [(75, 265, 108, 286), (76, 302, 109, 323), (76, 340, 109, 361), (76, 378, 109, 399), (77, 415, 110, 436), (77, 453, 110, 474), (78, 490, 111, 511), (78, 528, 111, 549), (79, 565, 112, 586), (79, 603, 112, 624), (80, 640, 113, 661), (80, 678, 113, 699), (80, 715, 113, 736)], [(90, 1504, 123, 1525), (91, 1541, 124, 1562), (91, 1579, 124, 1600)], [(87, 1279, 120, 1300), (88, 1316, 121, 1337), (88, 1354, 121, 1375)], [(1019, 704, 1052, 725), (794, 706, 827, 727)], [(1015, 366, 1048, 387), (790, 369, 823, 390)], [(1023, 1041, 1056, 1062), (797, 1044, 830, 1065)]], [[(674, 897, 707, 918), (675, 934, 708, 955), (674, 972, 707, 993), (675, 1009, 708, 1030), (674, 1047, 707, 1068), (674, 1084, 707, 1105), (674, 1122, 707, 1143), (674, 1160, 707, 1181), (674, 1197, 707, 1218), (674, 1235, 707, 1256), (674, 1273, 707, 1294), (674, 1310, 707, 1331), (674, 1348, 707, 1369), (674, 1385, 707, 1406), (674, 1422, 707, 1443), (674, 1459, 707, 1480), (674, 1497, 707, 1518), (673, 1535, 706, 1556), (674, 1573, 707, 1594), (673, 1610, 706, 1631), (673, 1647, 706, 1668), (673, 1685, 706, 1706), (673, 1722, 706, 1743), (673, 1760, 706, 1781), (673, 1798, 706, 1819)], [(71, 822, 104, 843), (71, 859, 104, 880), (71, 896, 104, 917), (71, 934, 104, 955), (71, 971, 104, 992), (71, 1009, 104, 1030), (71, 1047, 104, 1068), (70, 1084, 103, 1105), (70, 1122, 103, 1143), (70, 1159, 103, 1180)]], [[(80, 1492, 113, 1513), (79, 1529, 112, 1550), (79, 1567, 112, 1588), (79, 1605, 112, 1626)], [(77, 1755, 110, 1776), (77, 1792, 110, 1813), (77, 1830, 110, 1851), (76, 1867, 109, 1888), (77, 1904, 110, 1925)], [(85, 855, 118, 876), (85, 893, 118, 914), (84, 930, 117, 951), (84, 968, 117, 989), (84, 1005, 117, 1026), (84, 1042, 117, 1063), (83, 1080, 116, 1101), (83, 1117, 116, 1138)], [(82, 1267, 115, 1288), (82, 1305, 115, 1326), (82, 1342, 115, 1363)], [(800, 711, 833, 732), (1025, 713, 1058, 734)], [(91, 256, 124, 277), (90, 293, 123, 314), (90, 331, 123, 352), (90, 368, 123, 389), (89, 406, 122, 427), (89, 444, 122, 465), (89, 481, 122, 502), (88, 518, 121, 539), (88, 556, 121, 577), (88, 593, 121, 614), (87, 630, 120, 651), (87, 667, 120, 688), (86, 705, 119, 726)], [(803, 375, 836, 396), (1028, 377, 1061, 398)]], [[(80, 1491, 113, 1512), (80, 1528, 113, 1549), (79, 1566, 112, 1587)], [(78, 1716, 111, 1737), (77, 1754, 110, 1775), (77, 1791, 110, 1812), (76, 1829, 109, 1850), (76, 1866, 109, 1887)], [(87, 853, 120, 874), (87, 891, 120, 912), (86, 929, 119, 950), (86, 966, 119, 987), (85, 1004, 118, 1025), (85, 1041, 118, 1062), (84, 1079, 117, 1100), (84, 1116, 117, 1137)], [(82, 1266, 115, 1287), (82, 1304, 115, 1325), (82, 1342, 115, 1363)], [(801, 712, 834, 733), (1027, 714, 1060, 735)], [(94, 255, 127, 276), (94, 292, 127, 313), (93, 330, 126, 351), (92, 368, 125, 389), (92, 405, 125, 426), (92, 442, 125, 463), (91, 480, 124, 501), (91, 517, 124, 538), (91, 554, 124, 575), (90, 592, 123, 613), (90, 629, 123, 650), (90, 666, 123, 687), (89, 704, 122, 725)], [(806, 376, 839, 397), (1031, 379, 1064, 400)]], [[(678, 897, 711, 918), (678, 934, 711, 955), (678, 971, 711, 992), (678, 1009, 711, 1030), (678, 1046, 711, 1067), (678, 1084, 711, 1105), (678, 1121, 711, 1142), (678, 1159, 711, 1180), (678, 1196, 711, 1217), (678, 1234, 711, 1255), (679, 1271, 712, 1292), (678, 1309, 711, 1330), (678, 1347, 711, 1368), (679, 1383, 712, 1404), (678, 1421, 711, 1442), (679, 1458, 712, 1479), (679, 1496, 712, 1517), (679, 1533, 712, 1554), (679, 1570, 712, 1591), (679, 1608, 712, 1629), (679, 1645, 712, 1666), (679, 1683, 712, 1704), (679, 1720, 712, 1741), (679, 1758, 712, 1779), (679, 1795, 712, 1816)], [(75, 823, 108, 844), (75, 861, 108, 882), (75, 898, 108, 919), (75, 935, 108, 956)]], [[(679, 896, 712, 917), (679, 934, 712, 955), (679, 971, 712, 992), (679, 1009, 712, 1030), (679, 1046, 712, 1067), (679, 1084, 712, 1105), (679, 1121, 712, 1142), (679, 1159, 712, 1180), (679, 1197, 712, 1218), (679, 1234, 712, 1255), (678, 1272, 711, 1293), (678, 1309, 711, 1330), (679, 1347, 712, 1368), (679, 1383, 712, 1404), (678, 1421, 711, 1442), (678, 1458, 711, 1479), (678, 1496, 711, 1517), (678, 1533, 711, 1554), (678, 1571, 711, 1592), (678, 1608, 711, 1629), (678, 1646, 711, 1667), (678, 1683, 711, 1704), (678, 1721, 711, 1742), (678, 1758, 711, 1779), (678, 1796, 711, 1817)], [(76, 822, 109, 843), (76, 859, 109, 880), (76, 896, 109, 917), (76, 934, 109, 955), (76, 971, 109, 992), (76, 1009, 109, 1030), (76, 1046, 109, 1067), (76, 1084, 109, 1105), (76, 1121, 109, 1142), (76, 1159, 109, 1180)]], [[(1015, 370, 1048, 391), (790, 372, 823, 393)], [(75, 228, 108, 249), (75, 266, 108, 287), (76, 303, 109, 324)], [(1022, 1043, 1055, 1064), (796, 1045, 829, 1066)], [(1018, 706, 1051, 727), (793, 708, 826, 729)]], [[(675, 1724, 708, 1745), (675, 1761, 708, 1782), (675, 1798, 708, 1819)], [(74, 935, 107, 956), (74, 973, 107, 994), (74, 1010, 107, 1031), (74, 1048, 107, 1069), (74, 1085, 107, 1106), (74, 1122, 107, 1143), (73, 1160, 106, 1181), (74, 1197, 107, 1218), (73, 1235, 106, 1256), (73, 1273, 106, 1294), (73, 1310, 106, 1331), (73, 1348, 106, 1369), (73, 1385, 106, 1406), (73, 1422, 106, 1443), (73, 1460, 106, 1481), (73, 1497, 106, 1518), (72, 1535, 105, 1556), (72, 1573, 105, 1594), (72, 1610, 105, 1631), (72, 1647, 105, 1668), (72, 1685, 105, 1706), (72, 1722, 105, 1743), (72, 1759, 105, 1780), (72, 1797, 105, 1818), (72, 1835, 105, 1856)], [(677, 711, 710, 732), (677, 748, 710, 769), (677, 786, 710, 807), (678, 823, 711, 844), (677, 861, 710, 882), (677, 898, 710, 919), (677, 936, 710, 957), (677, 973, 710, 994), (677, 1011, 710, 1032), (677, 1048, 710, 1069), (677, 1085, 710, 1106), (677, 1123, 710, 1144), (677, 1161, 710, 1182)], [(676, 1311, 709, 1332), (676, 1349, 709, 1370), (676, 1386, 709, 1407), (676, 1423, 709, 1444), (676, 1461, 709, 1482), (676, 1498, 709, 1519), (676, 1536, 709, 1557), (676, 1573, 709, 1594)]], [[(678, 895, 711, 916), (678, 932, 711, 953), (678, 969, 711, 990), (678, 1007, 711, 1028), (678, 1044, 711, 1065), (678, 1082, 711, 1103), (678, 1119, 711, 1140), (678, 1157, 711, 1178), (678, 1194, 711, 1215), (678, 1232, 711, 1253), (678, 1269, 711, 1290), (678, 1307, 711, 1328), (678, 1344, 711, 1365), (678, 1381, 711, 1402), (678, 1419, 711, 1440), (679, 1456, 712, 1477), (678, 1494, 711, 1515), (679, 1531, 712, 1552), (678, 1569, 711, 1590), (679, 1606, 712, 1627), (678, 1644, 711, 1665), (679, 1681, 712, 1702), (679, 1719, 712, 1740), (679, 1756, 712, 1777), (679, 1794, 712, 1815)], [(75, 821, 108, 842), (75, 858, 108, 879), (75, 895, 108, 916), (75, 933, 108, 954)]], [[(677, 896, 710, 917), (677, 934, 710, 955), (677, 971, 710, 992), (677, 1009, 710, 1030), (677, 1046, 710, 1067), (677, 1084, 710, 1105), (677, 1122, 710, 1143), (676, 1159, 709, 1180), (677, 1196, 710, 1217), (676, 1234, 709, 1255), (676, 1272, 709, 1293), (676, 1309, 709, 1330), (676, 1347, 709, 1368), (676, 1384, 709, 1405), (676, 1422, 709, 1443), (676, 1459, 709, 1480), (676, 1496, 709, 1517), (675, 1534, 708, 1555), (676, 1571, 709, 1592), (675, 1609, 708, 1630), (675, 1646, 708, 1667), (675, 1684, 708, 1705), (675, 1721, 708, 1742), (675, 1759, 708, 1780), (675, 1797, 708, 1818)], [(75, 822, 108, 843), (74, 859, 107, 880), (74, 896, 107, 917), (74, 934, 107, 955), (74, 971, 107, 992), (74, 1009, 107, 1030), (74, 1046, 107, 1067)]], [[(674, 1718, 707, 1739), (674, 1756, 707, 1777), (673, 1793, 706, 1814)], [(73, 931, 106, 952), (74, 968, 107, 989), (73, 1006, 106, 1027), (73, 1043, 106, 1064), (73, 1081, 106, 1102), (73, 1118, 106, 1139), (73, 1155, 106, 1176), (73, 1193, 106, 1214), (73, 1230, 106, 1251), (72, 1268, 105, 1289), (72, 1305, 105, 1326), (72, 1343, 105, 1364), (72, 1380, 105, 1401), (72, 1417, 105, 1438), (72, 1454, 105, 1475), (72, 1492, 105, 1513), (72, 1529, 105, 1550), (72, 1566, 105, 1587), (72, 1604, 105, 1625), (72, 1641, 105, 1662), (71, 1679, 104, 1700), (72, 1716, 105, 1737), (71, 1754, 104, 1775), (71, 1791, 104, 1812), (71, 1829, 104, 1850)], [(676, 708, 709, 729), (677, 745, 710, 766), (677, 782, 710, 803), (676, 820, 709, 841), (676, 857, 709, 878), (676, 895, 709, 916), (676, 932, 709, 953), (676, 970, 709, 991), (676, 1007, 709, 1028), (676, 1044, 709, 1065), (676, 1082, 709, 1103), (676, 1119, 709, 1140), (676, 1156, 709, 1177)], [(675, 1307, 708, 1328), (675, 1344, 708, 1365), (675, 1381, 708, 1402), (675, 1418, 708, 1439), (675, 1456, 708, 1477), (675, 1493, 708, 1514), (675, 1530, 708, 1551), (674, 1568, 707, 1589)]], [[(83, 858, 116, 879), (83, 895, 116, 916), (83, 933, 116, 954), (83, 970, 116, 991), (83, 1007, 116, 1028), (83, 1045, 116, 1066), (83, 1082, 116, 1103), (83, 1120, 116, 1141)], [(82, 1495, 115, 1516), (82, 1532, 115, 1553), (82, 1570, 115, 1591), (82, 1607, 115, 1628)], [(83, 1270, 116, 1291), (83, 1308, 116, 1329), (83, 1345, 116, 1366)], [(85, 258, 118, 279), (84, 296, 117, 317), (85, 333, 118, 354), (84, 371, 117, 392), (84, 408, 117, 429), (84, 446, 117, 467), (84, 484, 117, 505), (84, 521, 117, 542), (84, 558, 117, 579), (84, 596, 117, 617), (84, 633, 117, 654), (83, 670, 116, 691), (84, 707, 117, 728)], [(797, 709, 830, 730), (1022, 709, 1055, 730)], [(796, 1008, 829, 1029), (1022, 1008, 1055, 1029)], [(798, 372, 831, 393), (1023, 373, 1056, 394)]], [[(75, 1232, 108, 1253), (75, 1269, 108, 1290), (74, 1307, 107, 1328), (74, 1344, 107, 1365), (75, 1381, 108, 1402), (75, 1419, 108, 1440), (75, 1456, 108, 1477), (75, 1493, 108, 1514), (75, 1531, 108, 1552), (75, 1568, 108, 1589)], [(74, 821, 107, 842), (74, 858, 107, 879), (74, 895, 107, 916), (74, 932, 107, 953), (74, 970, 107, 991), (74, 1007, 107, 1028), (74, 1045, 107, 1066)], [(677, 894, 710, 915), (677, 931, 710, 952), (677, 969, 710, 990), (677, 1007, 710, 1028), (677, 1044, 710, 1065), (677, 1081, 710, 1102), (677, 1119, 710, 1140), (677, 1156, 710, 1177), (678, 1193, 711, 1214), (677, 1231, 710, 1252), (677, 1269, 710, 1290), (677, 1306, 710, 1327), (678, 1343, 711, 1364), (678, 1380, 711, 1401), (677, 1418, 710, 1439), (678, 1455, 711, 1476), (678, 1493, 711, 1514), (678, 1530, 711, 1551), (678, 1568, 711, 1589), (678, 1606, 711, 1627), (678, 1643, 711, 1664), (678, 1680, 711, 1701), (678, 1718, 711, 1739), (678, 1756, 711, 1777), (678, 1793, 711, 1814)]]])
