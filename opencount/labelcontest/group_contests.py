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

    best_val = 1<<30, None, None
    for num_writeins in range(len(texts2)):
        rottexts2 = [texts2[i:-num_writeins]+texts2[:i]+texts2[-num_writeins:] for i in range(len(texts2)-num_writeins)]
        values = [(sum(row_dist(a,b) for a,b in zip(texts1, t2)),i) for i,t2 in enumerate(rottexts2)]
        #print values
        minweight,order = min(values)

        #print 'min', order, minweight

        if best_val[0] > minweight:
            #print "SET", minweight
            best_val = minweight, order, num_writeins
    #print "BEST:", best_val
    #print 'so should be equal'
    #print texts1
    #print texts2[best_val[1]:-best_val[2]]+texts2[:best_val[1]]+texts2[-best_val[2]:]

    reorder = range(len(texts1))[best_val[1]:-best_val[2]]+range(len(texts1))[:best_val[1]]+range(len(texts1))[-best_val[2]:]
    #print 'order1', reorder
    return float(best_val[0]+val)/size, zip(range(len(reorder)), reorder), best_val[2]
    """
    
    """
    matching = []
    weights = sorted([(row_dist(a,b),a,b) for a in texts1 for b in texts2])
    while texts1 != [] and texts2 != []:
        found = False
        for weight,a,b in weights:
            if a in texts1 and b in texts2:
                print 'w', weight, texts1.index(a), texts2.index(b)
                #print 'pair', a, b
                matching.append((ordering1[texts1.index(a)],
                                 ordering2[texts2.index(b)]))
                del ordering1[texts1.index(a)]
                del ordering2[texts2.index(b)]
                del texts1[texts1.index(a)]
                del texts2[texts2.index(b)]
                val += weight
                found = True
                break
        if not found:
            print "---- FAILURE"
            #print otexts1
            #print otexts2
            #print texts1
            #print texts2
            return 1<<30, None
    print "MATCHING", matching
    print "result weight", float(val)/size
    return float(val)/size, matching
    #"""

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
        self.writin_num = 0

    def all_children(self):
        res = [self]
        for child in self.children:
            res += child.all_children()
        return res

    def get_root(self):
        while self.parent != self.parent.parent:
            self.parent = self.parent.parent
        return self.parent
    
    def join(self, new_parent):
        if self.get_root() == new_parent.get_root():
            return

        root1 = self.parent
        root2 = new_parent.parent

        if root1.depth < root2.depth:
            root1.parent = root2
            root2.children.append(root1)
        elif root2.depth < root1.depth:
            root2.parent = root1
            root1.children.append(root1)
        else:
            root1.parent = root2
            root1.depth += 1
            root2.children.append(root1)
    
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

    contests = [Contest(contests_text, i) for i in range(len(contests_text))]
    for (k1,k2),d in diff:
        contests[k1].similarity[k2] = d
        contests[k2].similarity[k1] = d
    print "Created"
    for (k1,k2),d in diff:
        if d[0] < .1:
            contests[k1].join(contests[k2])
    print "Traverse"
    seen = {}
    res = []
    for contest in contests:
        if contest.get_root() in seen: continue
        seen[contest.get_root()] = True
        v = [x.cid for x in contest.get_root().all_children()]
        res.append([(contests_text[x][:2],contest.similarity[x][1]) for x in v])
        print res[-1]

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
    print "ARGUMENTS", (t, paths, giventargets, lang_map)
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


#equ_class([[(0, [(619, 285, 1113, 2647)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'E \xa4Av\xa4\xa4 ALEX Lsvnrr\nParty Prelerence; Democratic\nComputer Scientist/Engineer\n\n'), (True, u'E 0\u2022=<n.v mrz\nPany Preference; Republican\nDoctor/Atiorney/Busirmessxuornan\n\n'), (True, u'Z AL Rnuunzz\nParty Prelerenuez Republimn\nBusinessman\n\n'), (True, u'E DIRK ALl.EN Kouovnx\nParty Preference: Republi \xa4 n\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republi n\nRetired Administralion Direcfor\n\n'), (True, u'E MIKE sTR\xa4M|.1N<;\nParty Preference; Democratic\nConsumer Rights Attomey\n\n'), (True, u'\xa4 DIANE srEwARr\nParty Preference: Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'E NAK SHAH\nParty Preference: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NAcHuM SHIFREN\nPar1y Preference: Republican\nEducator/Author/Businessman\n\n'), (True, u'E DENNIS JACKSON\nParty Preference; Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucuzs\nParty Prelerence: Republican\nSmall Business Owner\n\n'), (True, u'\xa4 GREG c0NL0r~1\nParty Prelerencez Republican\nBusinessman/CPA\n\n'), (True, u'E JOHN B0RuFr\nParty Prelerencez Republican\nBusinessman\n\n'), (True, u'2 oscm ALEJANDR0 sRAuN\nParty Prslerenoe: Republican\nBusinessman/Rancher\n\n'), (True, u'\xa4 MARSHA Ferwumu\nParty Preference; Peace and Freedom\nRetired Teacher\n\n'), (True, u'D DIANNE \u2022=E1NsrE1N\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0u.EEN SHEA FERNALD\nParty Preference: Democratic\nMoiher/C0nsu|tanIIAr1isl\n\n'), (True, u'E EuzAsErH EMKEN\nParty Preference: Republican\nBusiness\xbbumnanIN<x1pro|it Executive\n\n'), (True, u'E KABIRUDDIN \xbb<AR\xa4M ALI\nPany Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wnnuxxms\nParty Preference: Republican\nBusiness Atiomey\n\n'), (True, u'E R0<;Eu0 T. cn.0RaA\nParty Preference; Republican\nGraduate Student/Businessman\n\n'), (True, u'\xa4 \xa40N Jr cnuwummu\nParty Preference. American Independen\nDoctor of Chiropractic\n\n')]), (0, [(1107, 285, 1598, 597)], [(False, u'\n'), (True, u'E ROBERT LAUTEN\nParty Preference: Republican\n\n'), (True, u'\xa4 Gm K, uGHn=00T\nParty Prererenw. Ubedarian\nRetired Nurse\n\n'), (True, u'Z\n\n')]), (0, [(1107, 591, 1598, 1096)], [(False, u'UNITED STATES REPRESENTATIVE\n3901 District\nVote lor One\n\n'), (True, u'\xa4 cmmne Mun.AmER\xa4\nParty Prelerence; None\nCommunity Volunteer\n\n'), (True, u'E .uAv cum\nParty Preference: Democratic\nBusinessman/School Bcardmember\n\n'), (True, u'D ED Rovcz\nParty Preference: Repubhcan\nU.S. Representative\n\n'), (True, u'Z\n\n')]), (0, [(1107, 1090, 1598, 1493)], [(False, u'STATE SENATOR\n29th Disfrict\nVote for Ons\n\n'), (True, u'\xa4 GREG DIAMOND\nParty Prelerencez Democratic\nWort<ers\u2018 Rights Attorney\n\n'), (True, u'E Roaenr "BOB" Hur=F\nParty Preference: Republican\nLawmakerlBusiness Owner\n\n'), (True, u'III\n\n')]), (0, [(1107, 1487, 1598, 1891)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n55\\h District\nVous for Ons\n\n'), (True, u'E cum HAGMAN\nParty Preference: Republican\nBusiness Owner/Assemblyman\n\n'), (True, u'Q GREGG D. \xbb=Rnc\xbb\u2014n.E\nParty Preference; Democratic\nSocial Worker\n\n'), (True, u'lj\n\n')]), (0, [(1107, 2188, 1598, 2532)], [(False, u'JudgaoI\u20221*\xa4aS\u2022.;>ori01C\xa4\xa4rt\nOfhcaN0.1\nV\xa4ts|0fOnc\n\n'), (True, u'E Euseme .uzHA\xbb<\nGeneral Practice Attorney\n\n'), (True, u'E mzaomxu .1.CHUANG\nJudge ol the Superior Court\n\n'), (True, u'III\n\n')])], [(1, [(614, 280, 1109, 2645)], [(False, u'UNITED STATES SENATOR\nVote for Ono\n\n'), (True, u'E \xa4Avn0 Auax uavrrr\nParty Preference: Democraiic\nComputer Scientist/Engineer\n\n'), (True, u'\xa4 0RLv TA|rz\nParty Preference: Repdzlium\nDoctor/Attorney/Bsnsinesswunarn\n\n'), (True, u'E AL Rmvnasz\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E DIRK ALLEN K0N0n\xa4n<\nParty Preference: Republican\nMBA Student\n\n'), (True, u'D DONALD KRAMPE\nParty Preterence: Republican\nRetired Administration Director\n\n'), (True, u'E Mme smnmuwc\nParty Prelerencez Democratic\nConsumer Rights Attorney\n\n'), (True, u'E DIANE STEWART\nParty Preference; Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'Z NAK slum\nParty Prelerence: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'E NAc+\u2022uM sHnr=REN\nParty Preference: Republican\nEducator/Autfror/Businessrnan\n\n'), (True, u'E DENNIS JAcr<s0N\nParty Prelerence: Republican\nAerospace General Manager\n\n'), (True, u'E DAN HUGHES\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'E GREG comow\nParty Preference: Republican\nBusinessman/CPA\n\n'), (True, u'E Jon-aw a0Ru\xbb=\xbb=\nPany Prelerenoe: Republi n\nBusinessman\n\n'), (True, u'Z 0scAR ALEJANDRO BRAuN\nParty Prelerencez Repubhcan\nBusinessman/Rancher\n\n'), (True, u'E MARsuA FEINLAND\nParty Preference; Peace and Freedom\nRetired Teacher\n\n'), (True, u'E \xa4nANNE Franwsrenw\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'\xa4 c0|.LEEr~1 SHEA FERNALD\nParty Preference: Democratic\nMother/Consultantlmtist\n\n'), (True, u'E EUZABETH EMKEN\nParty Preference: Republican\nBusinesswcerman/Nor1pr0Ii\\ Execume\n\n'), (True, u'E \u2022<Aa1Ru\xa4mN Kmnm ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'D Rncx wu.uAMs\nParty Prdecenoex Republican\nBusiness Attorney\n\n'), (True, u'E R0GEu0 T. c;|.0R|A\nParty Prelerencez Republican\nGraduate StudentIBusinessman\n\n'), (True, u'E DON J. GRUNDMANN\nParty Preference: American lndependem\nDoctor ol Chnropractic\n\n')]), (1, [(1103, 280, 1592, 593)], [(False, u'\n'), (True, u'\xa4 Romam |.AuTEN\nParty Prelercncez Repuxlican\n\n'), (True, u'\xa4 GAIL K u<;mi=00T\nParty Preterencei Libertarian\nRetired Nurse\n\n'), (True, u'III\n\n')]), (1, [(1103, 587, 1592, 1094)], [(False, u'UNITED STATES REPRESENTATIVE\n39th Dlstdci\nVqts Icr Ona\n\n'), (True, u'\xa4 \xa4\xb7MARra Mumwnenn\nParty Preference: None\nCommunaty Volunteer\n\n'), (True, u'E JAY cum\nParty Preference: Demouatic\nBusinessman/School Boardmember\n\n'), (True, u'\xa4 ED Rovcra\nParty Preference: Repubhmn\nU.5, Representative\n\n'), (True, u'[II\n\n')]), (1, [(1103, 1088, 1592, 1491)], [(False, u'STATE SENATOR\n29m District\nVob br Ona\n\n'), (True, u"E GREG DIAMOND\nParty Preterence: Democratic\nW0rkers' Rights Attorney\n\n"), (True, u'\xa4 ROBERT \xb7\xb7BOB\xb7 \u2022-\u2022ur=r=\nParty Preference: Republican\nLawmaker/Business Owner\n\n'), (True, u'Z\n\n')]), (1, [(1103, 1485, 1592, 1889)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n55\\h Distrid\nVous for One\n\n'), (True, u'\xa4 cum HAGrv1AN\nParty Preference: Repubhcan\nBusiness Owner/Assemblyman\n\n'), (True, u'E GREGG D. Fnncr-me\nParly Preference: Democratic\nSocial Worker\n\n'), (True, u'E\n\n')]), (1, [(1103, 2185, 1592, 2529)], [(False, u'JlK$\xb0O0\u2018IYBS|.Q0|i\xa2\xa5C(Xl\u2018{\n0||i0\xa4N0.1\nV0hf\xa4fOna\n\n'), (True, u'E EUGENE .nzm\xbb<\nGeneral Practace Attorney\n\n'), (True, u'E 0r;B0RAH J. cHuANG\nJudge ol me Superior Coun\n\n'), (True, u'EI\n\n')])], [(2, [(619, 278, 1113, 2647)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'\xa4 \xa4Av\u2022c> ALEX Lzzvm\nParty Preference: Democratic\nComputer Scieniist/Engineer\n\n'), (True, u'E 0RLv Tmz\nParty Preference; Republican\nDoctor/Att0rneylBusinessw\xa4\u2022\u2018nan\n\n'), (True, u'Q AL RAMrR&z\nParty Preference; Republican\nBusinessman\n\n'), (True, u'E \xa4u=<\xbb< ALLEN \u2022<0r~10\u2022>n<\nPany Preference: Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference; Republican\nRetired Adminisvation Direcwr\n\n'), (True, u'D MIKE sTRrMuNG\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'E muws STEWART\nParty Prelerencez Demcualic\nBusinesswcman/Finance Manager\n\n'), (True, u'E MAK sum\nParty Preterenoe; Democratic\nEnvironmental Health Consultant\n\n'), (True, u'D mcnum SHIFREN\nParty Prelerencez Republican\nEducator/Auttwor/Busirmessrnan\n\n'), (True, u'E DENNIS JACKSON\nParty Preference: Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucsuss\nParty Preference; Republkzan\nSmall Busmess Owner\n\n'), (True, u'Z GREG c0NL0N\nParty Prelerenoe: Republican\nBusinessman/CPA\n\n'), (True, u'\xa4 J0HN Bc>Rur=F\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E OSCAR ALEJANDRO awww\nParty Preference: Republican\nBus6nessmanIRa1cher\n\n'), (True, u'D Mmsm Famumu\nPa\u20221y Preference: Peace and Freedom\nRetired Teacher\n\n'), (True, u'Q DIANNE Fenwsmm\nParty Pretsrence: Democratic\nUnited States Senator\n\n'), (True, u'\xa4 c0LL&EN sun FERNAL0\nParty Preference; Democratic\nMother/Consultant/Artist\n\n'), (True, u'E E\xa4.nzABETH EMKEN\nParty Preierenoe: Republican\nBusinessw0manIN0npr0Ht Executive\n\n'), (True, u'E KABIRUDDIN KARIM ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wu.uAMs\nParty Preference: Republican\nBusiness Attorney\n\n'), (True, u'E Rocssuo T. cLc>RnA\nParty Preference: Repubhcan\nGraduate S!uden\\IBusinessman\n\n'), (True, u'D mom J. GRUNDMANN\nParty Preference: American lndepcnden\nDoctor of Chircpradic\n\n')]), (2, [(1107, 278, 1598, 595)], [(False, u'\n'), (True, u'\xa4 Roszm murem\nParty Preference. Republican\n\n'), (True, u'D Gm K. ucurroor\nParty Preference: Libertarian\nRetired Nurse\n\n'), (True, u'EI\n\n')]), (2, [(1107, 589, 1598, 1095)], [(False, u'UNITED STATES REPRESENTATIVE\n39th District\nVote Ior Ono\n\n'), (True, u'Q D\xb7MAR\xa4E MuLAmERn\nParty Preference: None\nCommunity Vulunteer\n\n'), (True, u'\xa4 Jn cwan\nParty Preisrencex Democratic\nBusinessman/Scmol Bnammember\n\n'), (True, u'E so Royce\nPady Preference: Republican\nU.S. Representative\n\n'), (True, u'IZ\n\n')]), (2, [(1107, 1089, 1598, 1492)], [(False, u'STATE SENATOR\n29Ih District\nVote Iur Ons\n\n'), (True, u"Q case DIAMOND\nParty Preference: Democratic\nW0rkers' Rights Attorney\n\n"), (True, u'D Rosaar \xb7\xa40\xa4\xb7 Hur:\nParty Preference: Republican\nLawmaksr/Business Ounev\n\n'), (True, u'Z\n\n')]), (2, [(1107, 1486, 1598, 1890)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n551h District\n[gm kx Ons\n\n'), (True, u'\xa4 cum HAGMAN\nParty Preference: Republi  n\nBusiness Oomev/Assemblyman\n\n'), (True, u'\xa4 emacs D \xbb=\xa4nc\xbb\u2014sLE\nParty Pretevence; Democratic\nSoda! Worker\n\n'), (True, u'IZ\n\n')]), (2, [(1107, 2187, 1598, 2530)], [(False, u'Judgs0Hh\xa4Sup\u2022ri0rCo\u2022n\nO|1lcoN0.1\nv\xa4n\xa4brOne\n\n'), (True, u'D EUGENE JIZHAK\nGeneral Practice Attorney\n\n'), (True, u'\xa4 DEBORAH J. cuumc\nJudge 0I the Superior Court\n\n'), (True, u'I2\n\n')])], [(3, [(620, 280, 1115, 2642)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'D ELIZABETH EMKEN\nParty Preierence Republican\nBusinessw0rnanINcnpr\xa4Ht Executive\n\n'), (True, u'E KABIRUDDIN \u2022<ARuv\xa4 ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E Rncx w|LuAMs\nParty Prelerencer Republican\nBusiness Aticmey\n\n'), (True, u'E R0<sEu0 T. G\xa4.0R\xa4A\nPany Preierence: Republican\nGraduate Student/Businessman\n\n'), (True, u'E 00N J. GRUNDMANN\nParty Preference: American lndependen\nDoctor 01 Chiropractic\n\n'), (True, u'D Roazm LAUTEN\nParty Preference; Republican\n\n'), (True, u'E ami. K. ucv-m=00T\nParty Preference: Libertarian\nRetired Nurse\n\n'), (True, u'E \xa4Avn\xa4 max Lsvnrr\nParty Prelerence; Democratic\nComputer S<:ientistlEngineer\n\n'), (True, u'\xa4 0RLY mrz\nParty Preference: Republi \xbb n\nDoctor/Attorney/Businessworrran\n\n'), (True, u'E AL RAMrREz\nParty Preference; Republican\nBusinessman\n\n'), (True, u'D 01RK ALLEN \u2022<0NOPn<\nParty Preference: Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republican\nRetired Administration Director\n\n'), (True, u'Q MIKE smnmuwc;\nParty Preference; Demouatic\nConsumer Rights Attomey\n\n'), (True, u'Q 0:ANE srew/mr\nParty Preference; Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'Q NAK sHA\xbb-1\nParty Prelevenoe; Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 mcnum sunmzm\nParty Preierence: Republican\nEducator/Author/Businessrnan\n\n'), (True, u'E DENNIS JAc\u2022<s0N\nParty Preference Republican\nAerospace General Manager\n\n'), (True, u'\xa4 DAN HUGHES\nParty Pralerence: Republican\nSmall Business Owner\n\n'), (True, u'E GREG <:0r~1L0N\nParty Preference: Republican\nBusmessman/CPA\n\n'), (True, u'E Jo}-iN BORUFF\nPariy Preference, Republican\nBusinessman\n\n'), (True, u'D OSCAR ALEJANDRO BRAUN\nParty Preference: Republican\nBusinessman/Rancher\n\n'), (True, u'\xa4 MARSHA Fznwumn\nParty Prelerence: Peace and Freedom\nRehred Teacher\n\n')]), (3, [(1109, 280, 1598, 590)], [(False, u'\n'), (True, u'Z DIANNE Ferwsrerw\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0Lu;&N SHEA FERNALD\nParty Preference: DemOcfa\\ic\nMother/Consultant/Artist\n\n'), (True, u'EI\n\n')]), (3, [(1109, 584, 1598, 1091)], [(False, u'UNITED STATES REPRESENTATIVE\n48th District\nVote lor One\n\n'), (True, u'\xa4 DANA R0:-aRABAcHeR\nPany Preference: Republican\nU.S. Representative\n\n'), (True, u'\xa4 ALAN SCHLAR\nParty Preference: None\nMarketing Sales Executive\n\n'), (True, u'D RON vARAs1\xb7EH\nPariy Preference: Democratic\nEngineer/Small Businessman\n\n'), (True, u'IZ]\n\n')]), (3, [(1109, 1085, 1598, 1829)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n72nd District\nVoin kx Ons\n\n'), (True, u'E \xb7rRAvls ALLEN\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'm ALBERT AYALA\nParty Preference: Democratic\nRetired Police Commander\n\n'), (True, u'E JOE \xa40v1NH\nParty Preierence: Democratic\nCity Commissioner/Businessperson\n\n'), (True, u'E LONc PHAM\nParty Prelerence: Republican\nMember, Orange County Board oi\nEducation\n\n'), (True, u'\xa4 mov EDGAR\nParty Preference: Republican\nBusinessman/Mayor\n\n'), (True, u'E\n\n')]), (3, [(1109, 2127, 1598, 2471)], [(False, u'Judge 0Hhs Superior Court\nOffice N0.1\nV0te|0rOne\n\n'), (True, u'E Eucseue .uz\u2022-\u2022A\xbb<\nGeneral Practice Attomey\n\n'), (True, u'm DEBORAH J, cHuAN<;\nJudge ul the Superior Court\n\n'), (True, u'III\n\n')])], [(4, [(621, 281, 1116, 2645)], [(False, u'UNITED STATES SENATOR\nVote lor One\n\n'), (True, u'Q Er.|zABErH EMKEN\nParty Preference Republican\nBusinesswoman/Nonproht Executive\n\n'), (True, u'E r<ABrRu\xa4\xa4\xa4N r<ARrM ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'Q Rncx w\xbbLuAMs\nParty Preference: Republican\nBusiness Allcmey\n\n'), (True, u'E ROGELIO T. G|.0R|A\nParty Preference; Republican\nGraduate Studen|IBusin$sma\\\n\n'), (True, u'D DON .1. <sRuN\xa4MANN\nParty Preierence: American Independen\nDodor 01 Chiropraciic\n\n'), (True, u'E Roarzm murem\nParty Preference; Republican\n\n'), (True, u'E GAIL K. uGHTF00T\nParry Preference: Libertarian\nRetired Nurse\n\n'), (True, u'D \xa4Avn0 Auax Levm\nParty Preference: Democratic\nComputer Scientist/Engineer\n\n'), (True, u'Z 0Ra.v mrz\nParty Preference: Repubhcan\nDoctorlAttomey/Businesswcrnan\n\n'), (True, u'E AL RAMnRE2\nParty Preference; Republican\nBusinessman\n\n'), (True, u'E max ALLEN \u2022<0N0Pn<\nParty Preference; Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republican\nRetired Adminnstration Director\n\n'), (True, u'E mma STRIMLING\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'E name sTEwARr\nParty Preference: Democratic\nBusinessw0manlFinanoe Manager\n\n'), (True, u'E NAK si-im\nParty Preference: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NAc\u2022-cum snnmew\nParty Preference: Republican\nEducator/Autfnor/Businessunan\n\n'), (True, u'E neuuns mcxm\nParty Prelerenuez Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucr-ues\nParty Prekzrenuez Republican\nSmall Business Owner\n\n'), (True, u'Z GREG c0N\xa4.0N\nParty Preierencex Republican\nBusinessman/CPA\n\n'), (True, u'\xa4 .10HN B0Rui=r\nPany Preference: Republican\nBusinessman\n\n'), (True, u'E 0scAR ALEJANDR0 BRAUN\nParty Preference: Republican\nBusinessmanIRancher\n\n'), (True, u'E MARSHA FEINLAND\nParty Preference: Peace and Freedom\nRetired Teacher\n\n')]), (4, [(1110, 281, 1601, 593)], [(False, u'\n'), (True, u'\xa4 DIANNE Fenwsrsm\nParty Prelerencez Democratic\nUnited States Senator\n\n'), (True, u'Q c0LLEEN SHEA FERNALD\nPany Preference: Democratic\nMother/Consultant/Artist\n\n'), (True, u'Z\n\n')]), (4, [(1110, 587, 1601, 1095)], [(False, u'UNITED STATES REPRESENTATIVE\n48Ih District\nVon Inr Ons\n\n'), (True, u'Q mm ROHRABACHER\nParty Preterencet Republican\nUS. Representative\n\n'), (True, u'\xa4 ALAN scH1.AR\nParty Prelerence None\nMarketing Sales Executive\n\n'), (True, u'E Rom vARAsm\xb71\nParty Preference: Democratic\nEngineer/Small Businessman\n\n'), (True, u'EI\n\n')]), (4, [(1110, 1089, 1601, 1832)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n72nd Disfrict\nVote for Ons\n\n'), (True, u'E TRAv\xa4s ALLEN\nParty Prelerervce: Republi n\nSmall Business Owner\n\n'), (True, u'\xa4 ALBERT AYALA\nParty Preference; Democratic\nRetired Police Commander\n\n'), (True, u'E Joe 00vmH\nParty Preference: Democratic\nCity Commissioner/Businessperson\n\n'), (True, u'\xa4 Lows PHAM\nParty Preferencet Republican\nMember, Orange County Board or\nEducation\n\n'), (True, u'\xa4 mov sucm\nParly Preference: Republican\nBusinessman/Mayor\n\n'), (True, u'EI\n\n')]), (4, [(1110, 2131, 1601, 2474)], [(False, u'Judgaoimcsenpovicrccnxt\nOffIt\xbbN0.1\nV\xa4m|or0ne\n\n'), (True, u'\xa4 EUGENE JIZHAK\nGeneral Practnce Attorney\n\n'), (True, u'2 \xa4za0RAH J. cnumcs\nJudge ol the Superior Court\n\n'), (True, u'E\n\n')])], [(5, [(139, 1152, 630, 1451)], [(False, u'PRESIDENT OF THE UNITED STATES\nVote lor Ons\n\n'), (True, u'E sTEwART ALEXANDER\n\n'), (True, u'E STEPHEN DURHAM\n\n'), (True, u'E ROSS c. \xb7R0c\u2022<Y\xb7 ANDERSON\n\n'), (True, u'EI\n\n')]), (5, [(624, 284, 1120, 2649)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'Q oscm ALEJANDRO BRAUN\nParty Preference: Republican\nBusinessman/Rancher\n\n'), (True, u'E MARs\xbb\u20141A FEnr~u.AN0\nParty Preference: Peace and Freedom\nRetired Teacher\n\n'), (True, u'\xa4 Dimmu; r=ErNsTr;n~1\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0LLsEN sa-on FERNALD\nParty Prelerenoez Democratic\nM0lherIC\xa4nsuItant/Artist\n\n'), (True, u'D ei.nzABem EMKEN\nParty Preference: Republican\nBusinesswornan/Nonprofit Executive\n\n'), (True, u'\xa4 KABIRUDDIN KARIM ALI\nParty Prelerencez Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wu.uAMs\nParty Preference: Republican\nBusiness Attomey\n\n'), (True, u'Z Roceuo TA c\xa4.0RnA\nParty Preference: Republican\nGraduate Student/Businessman\n\n'), (True, u'E DON J. GRUNDMANN\nParty Preference: American Independcn\nDoctor oi Chiropractic\n\n'), (True, u'Z ROBERT LAUTEN\nParty Prelerencez Republican\n\n'), (True, u'E GAnL K. uc}-nFOOT\nParty Preference; Libedarian\nRetired Nurse\n\n'), (True, u'E DAVID Auax Lzvnrr\nParty Preference: Democratic\nComputer SdentisvJEngineer\n\n'), (True, u'E 0RLY mrrz\nParty Prekzrencez Republican\nD0ct0rIAtt0rney/Busir\\essw0\u2022nan\n\n'), (True, u'Z AL RAMIREZ\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E DIRK ALLEN K0N0Pn<\nParty Preterence: Republi \xbb\xa2 n\nMBA Student\n\n'), (True, u'Z DONALD KRAMPE\nParty Preference: Republican\nRetired Administration Director\n\n'), (True, u'E Mme sTRnMuNG\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'Z umm; STEWART\nParty Prelevence: Democratic\nBusmesswuman/Finance Managa\n\n'), (True, u'\xa4 MAK sl-1AH\nParty Preterence: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NACHUM so-uFREN\nParty Preference: Republican\nEducator/Autr\u2022orIBusinessman\n\n'), (True, u'D DENNIS JACKSON\nParty Prelerence: Republican\nAerospace General Manager\n\n'), (True, u'D DAN Hucnss\nParty Prelerence: Republican\nSmall Business Owner\n\n')]), (5, [(1114, 284, 1604, 596)], [(False, u'\n'), (True, u'\xa4 GREG commu\nParty Preference: Republican\nBusmessman/CPA\n\n'), (True, u'\xa4 Jon-in \xa40Ru\xbb=\u2022=\nParty Preference: Republican\nBusinessman\n\n'), (True, u'IZ\n\n')]), (5, [(1114, 590, 1604, 1098)], [(False, u'UNITED STATES REPRESENTATIVE\n45th Distric!\nVon lor Ons\n\n'), (True, u'D .10HN wana\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'\xa4 sum-use KANG\nParty Preference: Democratic\nMayor ol Irvine\n\n'), (True, u'E .100-1N CAMPBELL\nParty Preference: Republican\nBusinessmanIU.S, Representative\n\n'), (True, u'EZI\n\n')]), (5, [(1114, 1092, 1604, 1494)], [(False, u'STATE SENATOR\n3701 District\nVots for Ons\n\n'), (True, u'E Mwu wALTERs\nParty Preference: Repubhcan\nBusinesssucrnan/Senator\n\n'), (True, u'E s\xb7rEvE voumc\nParty Preference: Democratic\nCivil Justice Attorney\n\n'), (True, u'EI\n\n')]), (5, [(1114, 1488, 1604, 1890)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n68th Distric!\n[ow br One\n\n'), (True, u'\xa4 cunnsnm AVALOS\nParty Prelerenoez Democratic\n\n'), (True, u'Q DONALD P. (Dom WAGNER\nParty Prelerence: Republican\nAssembly Member\n\n'), (True, u'III\n\n')]), (5, [(1114, 2188, 1604, 2532)], [(False, u'Judge ofthe Superior Court\nOfhce N0.1\nV\xa4tel\xa4rOne\n\n'), (True, u'\xa4 EUGENE Jnzwxx\nGeneral Practice Atiomey\n\n'), (True, u'E oEB0RAH J. cHuANG\nJudge of the Superior Court\n\n'), (True, u'III\n\n')])]])
