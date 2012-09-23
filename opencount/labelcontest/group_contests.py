from PIL import Image, ImageDraw
import os, sys
from random import random
sys.path.append('..')
try:
    from collections import Counter
except ImportError as e:
    from util import Counter
import multiprocessing as mp
import pickle
import itertools

black = 200

do_save = False
export = False

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

    result = [dfs_square([start]) for start in graph]
    result = [x for sublist in result for x in sublist]
    return list(set([x for x in result if x]))

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
            #print "Adding a contest", sq, inside, [area(intersect(sq, t)) for t in inside]
            contests.append(sq)
            targets = [x for x in targets if x not in inside]

    if targets != []:
        print "Was left with", targets
    keepgoing = True
    while keepgoing:
        keepgoing = False
        for target in giventargets:
            #print 'this target', target
            tomerge = [x for x in contests if intersect(x, target) == target]
            if len(tomerge) > 1:
                # Find the smallest subset to merge which overlap all targets in all contests.
                maxcontest = None
                must_include_targets = sum([[x for x in giventargets if intersect(c, x)] for c in tomerge],[])
                print 'must include', must_include_targets
                found = False
                for group_size in range(1,len(tomerge)+1):
                    if found: break
                    for comb in itertools.combinations(tomerge, group_size):
                        thiscontest = reduce(union, comb)
                        print 'this', thiscontest
                        print 'for each', [intersect(targ,thiscontest) for targ in must_include_targets]
                        if all(intersect(targ,thiscontest) for targ in must_include_targets):
                            print 'yes'
                            maxcontest = thiscontest
                            found = True
                            break
                print "MERGING", tomerge
                contests = [x for x in contests if x not in tomerge] + [maxcontest]
                keepgoing = True
                break
            elif len(tomerge) < 1:
                print "Target", target, "Not in any contest on ballot", name
    
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
    #exit(0)
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
    if not os.path.exists(sub):
        os.mkdir(sub)
    res = []
    #print "CONTESTS", contests
    for c in contests:
        #print "TOMAKE", c
        if not os.path.exists(os.path.join(sub, "-".join(map(str,c)))):
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
    tops = sorted([a[1]-u-10 for a in targets])+[d]
    if tops[0] > 0:
        tops = [0]+tops
    else:
        tops = [0,0]+tops[1:] # In case the top is negative.
    #print contest
    #print "USING", tops
    blocks = []
    for count,(upper,lower) in enumerate(zip(tops, tops[1:])):
        istarget = (count != 0)
        print upper, lower
        if upper == lower:
            blocks.append((istarget, ""))
            continue
        #print "POS", upper, lower
        #print len(cont_area[upper:lower])
        name = os.path.join(path, str(count)+".tif")
        if not os.path.exists(name):
            img = num2pil(cont_area[upper:lower])
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
            
    
    #print blocks
    return blocks

import editdist

def row_dist(a, b):
    v = editdist.distance(a.encode("ascii", "ignore"), 
                          b.encode("ascii", "ignore"))
    #print 'r', v, a == b
    return v
    """
    Compute the edit distance between two strings.
    """
    if a == b: return 0
    prev = None
    curr = range(len(b)+1)
 
    for i in range(len(a)):
        #print curr
        prev = curr
        curr = [0]*(len(b)+1)
        curr[0] = i+1
        for j in range(len(b)):
            curr[j+1] = min(prev[j+1] + 1,
                            curr[j] + 1,
                            prev[j] + (a[i] != b[j]))
    return curr[-1]


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

    if len(otexts1) != len(otexts2):
        print "Possible error: tried to compare distance of two contests with different number of targets."
        return 1<<30

    texts1 = [x for t,x in otexts1 if t]
    texts2 = [x for t,x in otexts2 if t]
    # Text associated with targets only
    ordering1 = range(len(texts1))
    ordering2 = range(len(texts2))
    size = sum(map(len,[x for _,x in otexts1]))+sum(map(len,[x for _,x in otexts2]))
    #print 'size', size
    if size == 0:
        print "Possible Error: A contest has no text associated with it"
        return 0, []

    titles1 = [x for t,x in otexts1 if not t]
    titles2 = [x for t,x in otexts2 if not t]
    val = sum(row_dist(*x) for x in zip(titles1, titles2))
    #print 'dist of titles is', val

    all_vals = []
    for num_writeins in range(len(texts2)):
        rottexts2 = [texts2[i:-num_writeins]+texts2[:i]+texts2[-num_writeins:] for i in range(len(texts2)-num_writeins)]
        values = [(sum(row_dist(a,b) for a,b in zip(texts1, t2)),i) for i,t2 in enumerate(rottexts2)]
        #print values
        minweight,order = min(values)

        #print 'min', order, minweight

        all_vals.append((minweight, order, num_writeins))
    #print "BEST:", best_val
    #print 'so should be equal'
    #print texts1
    #print texts2[best_val[1]:-best_val[2]]+texts2[:best_val[1]]+texts2[-best_val[2]:]
    all_vals = sorted(all_vals)
    res = {}
    best = 1<<30, None
    for weight,order,num_writeins in all_vals:
        lst = range(len(texts1))
        new_order = lst[order:-num_writeins]+lst[:order]+lst[-num_writeins:]
        if float(weight+val)/size < best[0]:
            best = float(weight+val)/size, num_writeins
        res[num_writeins] = (float(weight+val)/size,
                             zip(lst, new_order))
    return res, best

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

class Contest:
    def __init__(self, contests_text, cid):
        self.contests_text = contests_text
        self.cid = cid
        # CID -> [(distance, order, numwritein)]
        self.similarity = {}
        self.parent = self
        self.depth = 0
        self.children = []
        self.writein_num = 0

    def all_children(self):
        res = [self]
        for child in self.children:
            res += child.all_children()
        return res

    def get_root(self):
        while self.parent != self.parent.parent:
            self.parent = self.parent.parent
        return self.parent

    def dominating_set(self):
        root = self.get_root()
        children = root.all_children()
        conn = {}
        for c1 in children:
            lst = []
            for c2 in children:
                if c1.similarity[c2.cid][root.writein_num][0] < .1:
                    lst.append(c2.cid)
            conn[c1.cid] = lst
        conn = conn.items()
        rem = {}
        used = []
        while len(rem) != len(children):
            item = max(conn, key=lambda x: len(x[1]))
            used.append(item[0])
            for v in item[1]:
                rem[v] = True
            rem[item[0]] = True
            conn = [(k,[x for x in v if x not in rem]) for k,v in conn if k not in rem]
        print "SET", used

    def isClose(self, other, num_writein):
        group1 = self.all_children()
        group2 = other.all_children()
        best = 1<<30, None
        #print 'joining', len(group1), len(group2)
        for nwi in set([self.writein_num, other.writein_num, num_writein]):
            distance = 0
            for c1 in group1:
                for c2 in group2:
                    distance += c1.similarity[c2.cid][nwi][0]
            distance /= len(group1)*len(group2)
            #print nwi, distance
            if distance < best[0]:
                best = distance, nwi
        #print 'pick', best
        return best[0] < .2, best[1]
    
    def join(self, new_parent, num_writein, force=True):
        if self.get_root() == new_parent.get_root():
            return

        root1 = self.parent
        root2 = new_parent.parent

        if force:
            winum = num_writein
        else:
            close, winum = root1.isClose(root2, num_writein)
            if not close: return

        if root1.depth < root2.depth:
            root1.parent = root2
            root2.children.append(root1)
            root2.writein_num = winum
        elif root2.depth < root1.depth:
            root2.parent = root1
            root1.children.append(root1)
            root1.writein_num = winum
        else:
            root1.parent = root2
            root1.depth += 1
            root2.children.append(root1)
            root2.writein_num = winum
    
def do_group_pairing_map(data):
    lst = []
    for i,a,j,b in data:
        lst.append(((i,j),compare(a[2], b[2])))
    return lst

def group_by_pairing(contests_text):
    """
    Group contests together by pairing them one at a time.

    Currently this is very slow. It's going to run n^2 comparisons,
    and then do a linear scan through each of them to make the groups.
    """

    contests = [Contest(contests_text, i) for i in range(len(contests_text))]

    print "Linear Scan"
    contests_text = sorted(contests_text, key=lambda x: sum(len(v[1]) for v in x[2]))
    for i,(c1,c2) in enumerate(zip(contests_text, contests_text[1:])):
        data, (score,winum) = compare(c1[2], c2[2])
        if score < .1:
            contests[i].join(contests[i+1], winum, force=True)
    
    print len(contests)
    seen = {}
    for contest in contests:
        if contest.get_root() in seen: continue
        seen[contest.get_root()] = True

    print len(seen)
    print sum([len(x.all_children())**.5 for x in seen])
    
    
        
    print "Prepare"
    pool = mp.Pool(mp.cpu_count())
    args = [(i,cont1,j,cont2) for i,cont1 in enumerate(contests_text) for j,cont2 in enumerate(contests_text) if j <= i]
    sets = [[] for _ in range(mp.cpu_count())]
    for i,each in enumerate(args):
        sets[i%len(sets)].append(each)
    print "Start"
    res = pool.map(do_group_pairing_map, sets)
    print "Done"
    diff = {}
    for each in res:
        for k,v in each:
            diff[k] = v
    diff = sorted(diff.items(), key=lambda x: x[1][0])
    print "Finish"

    for (k1,k2),(dmap,best) in diff:
        contests[k1].similarity[k2] = dmap
        contests[k2].similarity[k1] = dmap
    print "Created"
    for (k1,k2),(dmap,best) in diff:
        if best[0] < .2:
            contests[k1].join(contests[k2], best[1])
    print "Traverse"
    seen = {}
    res = []
    for contest in contests:
        if contest.get_root() in seen: continue
        contest.get_root().dominating_set()
        seen[contest.get_root()] = True
        v = [x.cid for x in contest.get_root().all_children()]
        write = contest.get_root().writein_num
        res.append([(contests_text[x][:2],contest.similarity[x][write][1]) for x in v])
        print "HASHCODE", hash(str(sorted(res[-1])))

    return res
            
def equ_class(contests):
    #print "EQU", contests
    #print map(len, contests)
    #print contests
    contests = [x for sublist in contests for x in sublist]
    #print contests
    groups = first_pass(contests)
    # Each group is known to be different.
    result = []
    for group in groups:
        result += group_by_pairing(group)
        print "Finished one group"
    
    #print "RETURNING", result
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
            equal = [i for t in group for i,(_,bounding,_) in enumerate(ballot) if intersect(t, bounding) == t]
            equal_uniq = list(set(equal))
            #print equal_uniq
            merged = sum([ballot[x][2] for x in equal_uniq],[])
            new_ballot.append((ballot[equal[0]][0], [ballot[x][1] for x in list(set(equal))], merged))
        new_data.append(new_ballot)
    #print new_data
    return new_data

def extend_multibox(ballots, box1, box2, orders):
    ballot = ballots[box1[0]]
    txt1 = [x for x in ballot if x[:2] == box1][0]
    txt2 = [x for x in ballot if x[:2] == box2][0]
    txt = txt1[2]+txt2[2]
    res = []
    newgroup = []
    for bid,order in enumerate(orders):
        #print 'BID IS', bid
        for c1,c2 in order:
            t1 = [x for x in ballots[bid] if x[:2] == c1][0]
            t2 = [x for x in ballots[bid] if x[:2] == c2][0]
            if len(t1[2])+len(t2[2]) != len(txt1[2])+len(txt2[2]):
                continue
            #print '-'*30
            #print 'consec', c1, c2
            score, order = compare(txt, t1[2]+t2[2])
            if score < .2:
                #print "THEY ARE EQUAL"
                res.append((c1, c2))
                print 'txt', t1, t2
                newgroup.append(((c1[0], [c1[1], c2[1]], t1[2]+t2[2]), order))
    print "RESULT", res

    return res, newgroup

# Wrong: 39 42 65

def do_grouping(t, paths, giventargets, lang_map = {}):
    global tmp
    #print "ARGUMENTS", (t, paths, giventargets, lang_map)
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
    #print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets)

def find_contests(t, paths, giventargets):
    global tmp
    print "ARGS", (t, paths, giventargets)
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    args = [(f, sum(giventargets[i],[]), False) for i,f in enumerate(paths)]
    pool = mp.Pool(mp.cpu_count())
    ballots = map(extract_contest, args)
    #ballots = map(extract_contest, args)
    #print "RETURNING", ballots
    return ballots

def group_given_contests_map(arg):
    lang_map,giventargets,(i,(f,conts)) = arg
    print f
    im = load_num(f)
    lang = lang_map[f] if f in lang_map else 'eng'
    return ballot_preprocess(i, f, im, conts, sum(giventargets[i],[]), lang)
        
def group_given_contests(t, paths, giventargets, contests, lang_map = {}):
    global tmp
    #print "ARGUMENTS", (t, paths, giventargets, lang_map)
    #print 'giventargets', giventargets
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    #os.popen("rm -r "+tmp+"*")
    pool = mp.Pool(mp.cpu_count())
    args = [(lang_map,giventargets,x) for x in enumerate(zip(paths,contests))]
    ballots = pool.map(group_given_contests_map, args)
    #ballots = map(group_given_contests_map, args)
    #print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets)

def final_grouping(ballots, giventargets):
    print "RUNNING FINAL GROUPING"
    #pickle.dump((ballots, giventargets), open("/tmp/aaa", "w"))
    ballots = merge_contests(ballots, giventargets)
    print "NOW EQU CLASSES"
    #print ballots
    return equ_class(ballots)


if __name__ == "__main__":
    from labelcontest import LabelContest
    p = "../projects/my_yolo/"
    # Regroup the targets so that equal contests are merged.
    class FakeProj:
        target_locs_dir = p+"target_locations"
    class FakeMe(LabelContest):
        proj = FakeProj()
    fakeme = FakeMe(None, None)
    LabelContest.gatherData(fakeme)
    groupedtargets = fakeme.groupedtargets
    targets = []
    for bid,ballot in enumerate(groupedtargets):
        ballotlist = []
        for gid,targlist in enumerate(ballot):
            ballotlist.append([x[2:] for x in targlist])
        targets.append(ballotlist)

    internal = pickle.load(open(p+"contest_internal.p"))[2]
    final_grouping(internal, targets)
