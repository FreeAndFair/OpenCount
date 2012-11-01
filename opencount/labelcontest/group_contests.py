from PIL import Image, ImageDraw
import os, sys
import time
import random
sys.path.append('..')
try:
    from collections import Counter
except ImportError as e:
    from util import Counter
import multiprocessing as mp
import cPickle as pickle
import itertools
import time
from grouping.partask import do_partask

def pdb_on_crash(f):
    """
    Decorator to run PDB on crashing  
    """
    def res(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            import pdb as err_pdb
            err_pdb.post_mortem()
    return res

black = 200

do_save = True
do_test = True
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

    def dorem(dat, block, boxes, replacewith=False):
        remove = []
        for x,y in boxes:
            if (x,y-block) in boxes and (x,y+block) in boxes:
                if (x-block,y) in boxes and (x+block,y) in boxes:
                    remove.append((x,y))
        for x,y in remove:
            for dy in range(block):
                for dx in range(block):
                    dat[y+dy][x+dx] = replacewith


    dat = load_num(image)
    block = 40
    boxes = {}
    for y in range(0, len(dat)-block, block):
        for x in range(0, len(dat[y])-block, block):
            if sum(dat[y+dy][x+dx] < 240 for dy in range(0,block,4) for dx in range(0,block,4)) > block/4*block/4*3/10:
                lst = [dat[y+dy][x+dx] < 240 for dy in range(0,block) for dx in range(0,block)]
                if sum(lst) > block*block*7/10:
                    boxes[x,y] = True
    dorem(dat, block, boxes, replacewith=255)

    dat = [[x < black for x in y] for y in dat]
    block = 10
    boxes = {}
    for y in range(0,len(dat)-block, block):
        for x in range(0,len(dat[y])-block, block):
            if sum(dat[y+dy][x+dx] for dy in range(0,block,2) for dx in range(0,block,2)) > block/2*block/2*5/10:
                filled = sum(dat[y+dy][x+dx] for dy in range(block) for dx in range(block)) > block*block*9/10
                if filled:
                    boxes[x,y] = True

    dorem(dat, block, boxes, replacewith=255)
    
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

        lower = max(y-10,0)
        upper = min(y+10,height)

        q = [point[0]]
        x = point[1]-1
        while q and x > 0:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if lower <= dy+y < upper and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x -= 1
        l = x
        q = [point[0]]
        x = point[1]+1
        while q and x < width:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if lower <= dy+y < upper and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x += 1
        r = x

        return l,r

    def full_extend_ud_2(point):
        l,r = extend_lr(point)
        if r-l < 20: x = (l+r)/2
        else: x = point[1]
        point = (point[0],x)

        lower = max(x-10,0)
        upper = min(x+10,width)

        q = [point[1]]
        y = point[0]-1
        while q and y > 0:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if lower <= dx+x < upper and data[y][dx+x] < black]))
            #LST.extend([(x,y) for y in q])
            y -= 1
        u = y
        q = [point[1]]
        y = point[0]+1
        while q and y < height:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if lower <= dx+x < upper and data[y][dx+x] < black]))
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

@pdb_on_crash
def extend_to_line(lines, width, height):
    table = [[False]*(width+200) for _ in range(height+200)]
    for d,each in lines:
        if d == 'V':
            (l,u,r,d) = each
            for x in range(l,r):
                for y in range(u,d):
                    table[y][x] = True

    print "RECTANGLE FIX", len(lines)

    new = []
    for direc,each in lines:
        if direc == 'H':
            l,u,r,d = each
            if any(table[u][x] for x in range((l+9*r)/10,r)):
                new.append((direc,each))
            else:
                pos = [x for x in range((l+9*r)/10,min(r+(r-l)/2,width)) if table[u][x]]
                if len(pos):
                    new.append((direc, (l,u,pos[0]+(pos[0]-l)/10,d)))
        else:
            new.append((direc,each))

    print len(new)

    return new


def to_graph(lines, width, height, minsize):
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
            table = [[None]*(width+200) for _ in range(height+200)]
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
    print list(sorted([area(x[1]) for x in lines]))
    print minsize
    lines = [x for x in lines if x[1][2]-x[1][0] > width/10 or x[1][3]-x[1][1] > height/30]
    print "THERE ARE END", len(lines)

    lines = extend_to_line(lines, width, height)
    
    vertexes = dict((x, []) for _,x in lines)

    boxes = []
    for way1,line1 in lines:
        for way2,line2 in lines:
            if way1 != way2:
                if intersect(line1, line2):
                    boxes.append(intersect(line1, line2))
                    vertexes[line1].append(line2)
    print 'finished', len(str(vertexes)), len(boxes)
    return boxes,dict((k,v) for k,v in vertexes.items() if v != [])

def find_squares(graph, minarea):
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
                tores = intersect(union(stack[0],stack[2]), union(stack[1],stack[3]))
                if area(tores) > minarea:
                    return [tores]
            return [None]
        res = []
        for vertex in graph[stack[-1]]:
            if vertex in stack: continue
            res += dfs_square(stack+[vertex], debug)
        return res

    #result = [dfs_square([start]) for start in graph]
    #result = [x for sublist in result for x in sublist]
    #return list(set([x for x in result if x]))
    result = {}
    for i,start in enumerate(graph):
        print 'on', i, 'of', len(graph)
        for each in dfs_square([start]):
            if each:
                result[each] = True
    return result.keys()

def area(x): 
    if x == None: return 0
    return (x[2]-x[0])*(x[3]-x[1])

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

    def samecolumn(a, b):
        if (abs(a[0]-b[0]) < 10 or abs(a[2]-b[2])<10):
            if abs((a[0]+a[2])/2-(b[0]+b[2])/2) < 100:
                return True
        return False
        
    if len(contests) > 2*len(giventargets)/3:
        equivs = list(contests)
        prev = 0
        print "start", contests
        while prev != len(equivs):
            prev = len(equivs)
            new = []
            skip = {}
            for a in equivs:
                if a in skip: continue
                print "On ", a
                found = None
                for b in equivs:
                    if a == b: continue
                    if abs(b[3]-a[1]) < 30 and samecolumn(a, b):
                        print 'case 2', b
                        found = b
                        break
                if found != None:
                    print 'merge'
                    new.append(union(a, found))
                    skip[found] = True
                else:
                    new.append(a)
            equivs = []
            while new:
                s = [x for x in new if intersect(x,new[0]) and samecolumn(x, new[0])]
                equivs.append(reduce(union,s))
                new = [x for x in new if x not in s]
                
            print "RES", len(equivs), equivs

        contests = equivs
    
    #print "C", contests
    for cont in contests:
        if export:
            im = img.crop(cont)
            cname = tmp+"/"+str(sum(im.histogram()[:100]))+".png"
            im = img.crop(cont)
            im.save(cname)

    if do_save or do_test:
        new = Image.new("RGB", img.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for box in contests:
            c = (int(random.random()*200), int(random.random()*155+100), int(random.random()*155+100))
            imd.rectangle(box, fill=c)
        print "GIVEN", giventargets
        for box in giventargets:
            imd.rectangle(box, fill=(255,0,0))
        new.save(tmp+"/"+name+"-fullboxed.png")

    return contests

    #print targets, contests
    #os.popen("open tmp/*")
    #exit(0)
        

def extract_contest(args):
    try:
        return extract_contest_2(args)
    except:
        print "Fail on", args[0]
        print "Fail on", args[0]
        print "Fail on", args[0]
        print "Fail on", args[0]
def extract_contest(args):
    if len(args) == 2:
        image_path, giventargets = args
        returnimage = True
    elif len(args) == 3:
        image_path, giventargets, returnimage = args
    else:
        raise Error("Wrong number of args")

    now = time.time()

    print len(giventargets), giventargets

    print "processing", image_path
    data = load_threshold(image_path)
    #data = load_num(image_path)
    print 'loaded'
    lines = find_lines(data)
    lines += [('V', (len(data[0])-20, 0, len(data[0]), len(data)))]
    #print "calling with args", lines, len(data[0]), len(data), max(giventargets[0][2]-giventargets[0][0],giventargets[0][3]-giventargets[0][1])
    boxes, graph = to_graph(lines, len(data[0]), len(data), area(giventargets[0])**.5)
    print 'tograph'
    squares = find_squares(graph, area(giventargets[0]))
    print 'findsquares'
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
            c = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            imd.rectangle((l,u,r,d), fill=c)
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-box.png")

    if do_save or export or do_test:
        loadedimage = load_pil(image_path)
    else:
        loadedimage = None

    print "GET ARG", image_path, image_path.split("/")[-1]

    print len(giventargets), giventargets

    final = do_extract(image_path.split("/")[-1], 
                       loadedimage, squares, giventargets)
    #os.popen("open tmp/*")
    #exit(0)
    
    print "Took", time.time()-now

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
    cont_area = None
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
        name = os.path.join(path, str(count)+".tif")
        if os.path.exists(name+".txt"):
            txt = open(name+".txt").read().decode('utf8')
            if txt != '':
                print 'Found'
                blocks.append((istarget, txt))
                continue
            else:
                print 'Empty'
        #print "POS", upper, lower
        #print len(cont_area[upper:lower])
        if not os.path.exists(name):
            if cont_area == None: cont_area = load_num(pilimg=num2pil(image).crop((l+10,u+10,r-10,d-10)))
            img = num2pil(cont_area[upper:lower])
            img.save(name)
        
        txt = ""
        for iternum in range(3): # try 3 times
            if txt != "": continue
            os.popen("tesseract %s %s -l %s"%(name, name, lang))
            print 'Invoke tesseract'
            time.sleep(max((iternum-1)*.1,0))
            if os.path.exists(name+".txt"):
                print 'DONE'
                txt = open(name+".txt").read().decode('utf8')
                break

        if os.path.exists(name+".txt"):
            blocks.append((istarget, txt))
        else:
            print "-"*40
            print "OCR FAILED"
            print name
            print path
            print contest
            print lang
            print count, upper, lower
            print "-"*40
            blocks.append((istarget, ""))
            
    
    #print blocks
    return blocks

#import editdist
try:
    from Levenshtein import distance
except:
    print "Error: Edit Distance module not loaded"
    if __name__ != '__main__':
        print "Exiting"
        exit(1)

def row_dist(a, b):
    if type(a) == unicode or type(b) == unicode:
        return distance(unicode(a), unicode(b))
    else:
        return distance(a, b)
    #v = editdist.distance(a.encode("ascii", "ignore"), 
    #                      b.encode("ascii", "ignore"))
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
def compare(otexts1, otexts2, debug=False):
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

    def fixup(s):
        words = s.split()
        found = 0
        for item,sep in [('Party Preference: Republican', 3), ('Party Preference: Democratic', 3),
                         ('MEMBER OF THE STATE ASSEMBLY', 5)]:
            for i in range(len(words)-(sep-1)):
                combined = " ".join(words[i:i+sep])
                if abs(len(combined)-len(item)) < len(item)/10:
                    if row_dist(combined, item) < len(item)/5:
                        words[i:i+sep] = item.split(" ")
                        found += len(item)
        return " ".join(words), found

    texts1, founds1 = zip(*[fixup(x) for t,x in otexts1 if t])
    texts2, founds2 = zip(*[fixup(x) for t,x in otexts2 if t])
    # Text associated with targets only
    ordering1 = range(len(texts1))
    ordering2 = range(len(texts2))
    size = sum(map(len,[x for _,x in otexts1]))+sum(map(len,[x for _,x in otexts2]))-sum(founds1)-sum(founds2)
    #print 'size', size
    if size == 0:
        print "Possible Error: A contest has no text associated with it"
        return [(1<<30,(0,0,0)) for _ in range(len(texts1))], (1<<30, 0)

    titles1 = [x for t,x in otexts1 if not t]
    titles2 = [x for t,x in otexts2 if not t]
    val = sum(row_dist(*x) for x in zip(titles1, titles2))
    #print 'dist of titles is', val

    all_vals = []
    for num_writeins in range(len(texts2)):
        rottexts2 = [[texts2[i] for _,i in get_order(len(texts2),order,num_writeins)] for order in range(len(texts2))]
        values = [(sum(row_dist(a,b) for a,b in zip(texts1, t2)),i) for i,t2 in enumerate(rottexts2)]
        if debug:
            print "DEBUG", size, size-sum(map(len,titles1))-sum(map(len,titles2))
            print num_writeins
            print [([row_dist(a,b) for a,b in zip(texts1, t2)],i) for i,t2 in enumerate(rottexts2)]
            print map(len,texts1), map(len,texts2)
            print min(values)
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
        if float(weight+val)/size < best[0]:
            best = float(weight+val)/size, num_writeins
        res[num_writeins] = (float(weight+val)/size,
                             (len(texts1), order, num_writeins))
    return [x[1] for x in sorted(res.items())], best

def get_order(length, order, num_writeins):
    lst = range(length)
    if num_writeins == 0:
        new_order = lst[order:]+lst[:order]
    else:
        new_order = lst[order:-num_writeins]+lst[:order]+lst[-num_writeins:]
    return list(zip(lst, new_order))

def first_pass(contests, languages):
    """
    Split a set of contests in to a set of sets, where each
    set contains the same number of voting targets of the same language.
    """
    ht = {}
    i = 0
    for each in contests:
        key = (len(each[2]), None if each[0] not in languages else languages[each[0]])
        if key not in ht: ht[key] = []
        ht[key].append(each)
    return [x for x in sorted(ht.items())]

class Contest:
    def __init__(self, contests_text, cid, const=.2):
        self.contests_text = contests_text
        self.cid = cid
        self.const = const
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

    def is_close(self, other, num_writein):
        group1 = self.all_children()
        group2 = other.all_children()
        best = 1<<31, None
        #print 'joining', len(group1), len(group2)
        for nwi in set([self.writein_num, other.writein_num, num_writein]):
            distance = 0
            for c1 in group1:
                for c2 in group2:
                    if c2.cid not in c1.similarity:
                        distance += 1
                    else:
                        distance += c1.similarity[c2.cid][nwi][0]
            distance /= len(group1)*len(group2)
            #print nwi, distance
            if distance < best[0]:
                best = distance, nwi
        #print 'pick', best
        return best[0] < self.const, best[1]
    
    def join(self, new_parent, num_writein):
        if self.get_root() == new_parent.get_root():
            return

        root1 = self.parent
        root2 = new_parent.parent

        close, winum = root1.is_close(root2, num_writein)
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
    
def do_group_pairing_map(args):
    global tmp

    args = args[0]
    #for k,v in list(zip(globals(), [type(v) for k,v in globals().items()])):
    #    print k,v

    items, contests_text = args
    #out = open(tmp+"/group_dump/"+str(items[0]), "w")
    x = 0
    for i in items:
        lst = []
        for j in range(len(contests_text)):
            if x%10000 == 0: print x
            x += 1
            #print ((i,j),compare(contests_text[i][2], contests_text[j][2]))
            lst.append(((i,j),compare(contests_text[i][2], contests_text[j][2])))
        #out.write("\n".join(map(str,lst))+"\n")
        pickle.dump(lst, open(tmp+"/group_dump/"+str(i), "w"))
    #out.close()
    return []

def group_by_pairing(contests_text, CONST):
    """
    Group contests together by pairing them one at a time.

    Currently this is very slow. It's going to run n^2 comparisons,
    and then do a linear scan through each of them to make the groups.
    """
    global tmp

    contests = [Contest(contests_text, i, CONST) for i in range(len(contests_text))]

    #args = [(i,cont1,j,cont2) for i,cont1 in enumerate(contests_text) for j,cont2 in enumerate(contests_text)]


    #"""
    if not os.path.exists(tmp+"/group_dump"):
        os.mkdir(tmp+"/group_dump")
    else:
        os.popen("rm "+tmp+"/group_dump/*")

    """
    print "Prepare"
    pool = mp.Pool(mp.cpu_count())
    print "Start"
    data = [[] for _ in range(mp.cpu_count())]
    for i in range(len(contests_text)):
        data[i%len(data)].append(i)
    print "GO UP TO", (len(contests_text)**2)/mp.cpu_count()
    data = [(x, contests_text) for x in data]
    pool.map(do_group_pairing_map, data)
    pool.close()
    pool.join()
    #"""
    data = [[] for _ in range(mp.cpu_count())]
    for i in range(len(contests_text)):
        data[i%len(data)].append(i)
    print "GO UP TO", (len(contests_text)**2)/mp.cpu_count()
    data = [(x, contests_text) for x in data]
    do_partask(do_group_pairing_map, data, N=mp.cpu_count())
    

    diff = {}
    print len(contests_text)
    for i in range(len(contests_text)):
        if i%100 == 0:
            print 'load', i
        d = pickle.load(open(tmp+"/group_dump/"+str(i)))
        for k,v in d:
            diff[k] = v
        
    print "Done"
    diff = sorted(diff.items(), key=lambda x: x[1][1][0])
    print len(diff)
    #print diff[0]

    for (k1,k2),(dmap,best) in diff:
        contests[k1].similarity[k2] = dmap
    print "Created"
    for (k1,k2),(dmap,best) in diff:
        if best[0] > CONST: continue
        if k1 == k2: continue
        contests[k1].join(contests[k2], best[1])
        #print 'join', contests_text[k1][0], contests_text[k2][0]
        #print 'data', contests[k1].writein_num, contests[k2].writein_num
        #print contests_text[k1][2][1]
        #print contests_text[k2][2][1]
    print "Traverse"
    seen = {}
    res = []
    for contest in contests:
        #print 'try', contest.cid,
        contest = contest.get_root()
        if contest in seen: continue
        #print "SEE", contest
        #contest.dominating_set()
        seen[contest] = True
        v = [x.cid for x in contest.all_children()]
        #print "CHILDREN", v
        write = contest.writein_num
        #print "FOR THIS GROUP", write
        if contest.cid in contest.similarity:
            this = [(contests_text[contest.cid][:2],get_order(*contest.similarity[contest.cid][write][1]))]
        else:
            v = []
            l = range(len(contests_text[contest.cid][:2])-1)
            this = [(contests_text[contest.cid][:2],zip(l,l))]
        #print "Base"
        #print list(enumerate(contests_text[contest.cid][2][1:]))
        for x in v:
            if x == contest.cid: continue
            #print contest.similarity[x]
            #print contest.similarity[x][write][1], get_order(*contest.similarity[x][write][1])
            #print "This", list(enumerate(contests_text[x][2][1:]))
            this.append((contests_text[x][:2],get_order(*contest.similarity[x][write][1])))
        #print this
        res.append(this)
    print map(len,res)
    return res

def full_group(contests_text, key):
    print "Linear Scan"

    if key[1] == 'eng':
        CONST = .2
    elif key[1] == 'spa':
        CONST = .2
    elif key[1] == 'vie':
        CONST = .2
    elif key[1] == 'kor':
        CONST = .3
    elif key[1] == 'chi_sim':
        CONST = .3
    
    debug=[]

    contests_text = sorted(contests_text, key=lambda x: sum(len(v[1]) for v in x[2]))
    joins = dict((i,[]) for i in range(len(contests_text)))
    for offset in range(1,2):
        for i,(c1,c2) in enumerate(zip(contests_text, contests_text[offset:])):
            data, (score,winum) = compare(c1[2], c2[2])
            debug.append((score,(c1[2][0], c2[2][0])))
            if score < CONST/2:
                #print 'merged', c1[2], c2[2]
                joins[i].append(i+offset)
                joins[i+offset].append(i)
    def mylen(l):
        return sum(2 if ord(x)>512 else 1 for x in l)

    #for each in sorted(debug):
    #    print each[0]
    #    s1 = each[1][0][1].split("\n")
    #    s2 = each[1][1][1].split("\n")
    #    #print s1, s2
    #    s1 = [x+"."*(max(map(mylen,s1))-mylen(x)) for x in s1]
    #    print "\n".join([a+"  |  "+b for a,b in zip(s1,s2)])

    seen = {}
    exclude = {}
    for i in joins:
        if i in seen: continue
        items = dfs(joins, i)
        first = min(items)
        for each in items: seen[each] = True
        for each in items:
            if first != each:
                exclude[each] = first
    

    #print sorted(exclude.items())

    new_indexs = [x for x in range(len(contests_text)) if x not in exclude]
    new_contests = [contests_text[x] for x in new_indexs]

    print "Of sizes", len(contests_text), len(new_contests)
    #for x in new_contests[::100]:
    #    print x
    newgroups = []
    STEP = 1000
    print "Splitting to smaller subproblems:", len(new_contests)/STEP
    for iternum in range(0,len(new_contests),STEP):
        print "SUBPROB", iternum/STEP
        newgroups += group_by_pairing(new_contests[iternum:min(iternum+STEP, len(new_contests))], CONST)

    mapping = {}
    for i,each in enumerate(newgroups):
        for item in each:
            mapping[item[0][0],tuple(item[0][1])] = i
    #print "mapping", mapping

    for dst,src in exclude.items():
        #print "Get", dst, "from", src
        bid,cids = contests_text[src][:2]
        index = mapping[bid,tuple(cids)]
        find = newgroups[index][0][0]
        text = [text for bid,cid,text in contests_text if (bid,cid) == find][0]
        data,(score,winum) = compare(text, contests_text[dst][2])
        newgroups[index].append((contests_text[dst][:2], get_order(*data[winum][1])))
    
    #print "SO GET"
    #print sorted(map(hash,map(str,map(sorted,groups))))
    #print sorted(map(hash,map(str,map(sorted,newgroups))))

    return newgroups
    

            
def equ_class(contests, languages):
    #print "EQU", contests
    #print map(len, contests)
    #print contests
    contests = [x for sublist in contests for x in sublist]
    #print contests
    groups = first_pass(contests, languages)
    # Each group is known to be different.
    result = []
    print "Go up to", len(groups)
    for i,(key,group) in enumerate(groups):
        print "-"*50
        print "ON GROUP", i, key, len(group)
        print "-"*50
        result += full_group(group, key)
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

def do_extend(args):
    txt,c1,c2,t1,t2 = args
    data, (score, winum) = compare(txt, t1[2]+t2[2])
    if score < .2:
        #print "THEY ARE EQUAL"
        res = (c1, c2)
        #print 'txt', t1, t2
        newgroup = ((c1[0], [c1[1], c2[1]], t1[2]+t2[2]), get_order(*data[winum][1]))
        return res, newgroup
    return None
        

def extend_multibox(ballots, box1, box2, orders):
    ballot = ballots[box1[0]]
    txt1 = [x for x in ballot if x[:2] == box1][0]
    txt2 = [x for x in ballot if x[:2] == box2][0]
    txt = txt1[2]+txt2[2]
    res = []
    newgroup = []

    tocompare = []
    for bid,order in enumerate(orders):
        #print 'BID IS', bid
        for c1,c2 in order:
            t1 = [x for x in ballots[bid] if x[:2] == c1][0]
            t2 = [x for x in ballots[bid] if x[:2] == c2][0]
            if len(t1[2])+len(t2[2]) != len(txt1[2])+len(txt2[2]):
                continue
            tocompare.append((txt,c1,c2,t1,t2))
    pool = mp.Pool(mp.cpu_count())
    res = pool.map(do_extend, tocompare)
    pool.close()
    pool.join()
    print "RESULT", res
    res = [x for x in res if x != None]
    res, newgroup = zip(*res)
    print "RESULT", res
    print "NEWGROUP", newgroup

    return res, newgroup

# Wrong: 39 42 65

def do_grouping(t, paths, giventargets, lang_map = {}):
    global tmp
    #print "ARGUMENTS", (t, paths, giventargets, lang_map)
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp.replace(" ", "\\ ")+"*")
    ballots = []
    for i,f in enumerate(paths):
        print f
        im, contests = extract_contest((f, sum(giventargets[i],[])))
        lang = lang_map[f] if f in lang_map else 'eng'
        get = ballot_preprocess(i, f, im, contests, sum(giventargets[i],[]), lang)
        ballots.append(get)
    #print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets)

@pdb_on_crash
def find_contests(t, paths, giventargets):
    """
    Input:
        str T:
        list PATHS:
        list GIVENTARGETS: G[i][j][k] := k-th target of j-th contest of i-th ballot.
    """
    global tmp
    #print "ARGS", (t, paths, giventargets)
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp.replace(" ", "\\ ")+"*")
    args = [(f, sum(giventargets[i],[]), False) for i,f in enumerate(paths)]
    args = [args[0]]
    #args = [x for x in args if x[0] == "santacruz/DNPP_VBM/DNPP_VBM_00015-0.png"]
    #args = [x for x in args if 'DEM_PCT_00004-0.png' in x[0]]
    pool = mp.Pool(mp.cpu_count())
    ballots = pool.map(extract_contest, args)
    pool.close()
    pool.join()
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
    #print lang_map
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    #os.popen("rm -r "+tmp.replace(" ", "\\ ")+"*")
    pool = mp.Pool(mp.cpu_count())
    args = [(lang_map,giventargets,x) for x in enumerate(zip(paths,contests))]
    ballots = pool.map(group_given_contests_map, args)
    pool.close()
    pool.join()
    #ballots = map(group_given_contests_map, args)
    #print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets, paths, lang_map)

def final_grouping(ballots, giventargets, paths, languages):
    lookup = dict((x,i) for i,x in enumerate(paths))
    languages = dict((lookup[k],v) for k,v in languages.items())
    print "RUNNING FINAL GROUPING"
    #pickle.dump((ballots, giventargets), open("/tmp/aaa", "w"))
    ballots = merge_contests(ballots, giventargets)
    print "NOW EQU CLASSES"
    #print ballots
    return equ_class(ballots, languages)

def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. Does an inplace sort.
  From:
      http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 

import re
import csv

class ThreadDoInferContests:
    def __init__(self, queue, job_id, proj, *args, **kwargs):
        self.job_id = job_id
        self.queue = queue
        self.proj = proj

    def extract_data(self):
        """
        Stolen from labelcontest.py.

        This should be removed in favor of taking the data from
        this panel directly, instead of loading from the file.
        """
        res = []
        dirList = []
        for root,dirs,files in os.walk(self.proj.target_locs_dir):
            sort_nicely(files) # Fixes Marin ordering.
            for each in files:
                if each[-4:] != '.csv': continue
                gr = {}
                name = os.path.join(root, each)
                for i, row in enumerate(csv.reader(open(name))):
                    if i == 0:
                        # skip the header row, to avoid adding header
                        # information to our data structures
                        continue
                    # If this one is a target, not a contest
                    if row[7] == '0':
                        if row[8] not in gr:
                            gr[row[8]] = []
                        # 2,3,4,5 are left,up,width,height but need left,up,right,down
                        gr[row[8]].append((int(row[2]), int(row[3]), 
                                           int(row[2])+int(row[4]), 
                                           int(row[3])+int(row[5])))
                    r = row[0].replace("/media/data1/audits2012_straight/santacruz/blankballots/", "santacruz/")
                    if r not in dirList:
                        dirList.append(r)
                if gr.values() != []:
                    res.append(gr.values())
        #for a,b in zip(dirList, res):
        #    print a,b
        #print res
        #print dirList
        return res, dirList
        
    def run(self):
        # Do fancy contest-inferring computation
        data, files = self.extract_data()
        bboxes = dict(zip(files,find_contests(self.proj.ocr_tmp_dir, files, data)))
        # Computation done!
        self.queue.put(bboxes)
        self.proj.infer_bounding_boxes = True
        print "AND I SEND THE RESUTS", bboxes


if __name__ == "__main__":
    _, paths, them = pickle.load(open("marin_contest_run"))
    paths = [x.replace("/media/data1/audits2012_straight/marin/blankballots", "marin") for x in paths]
    find_contests("tmp", paths, them)
    exit(0)


tmp = "tmp"

"""
if __name__ == "__main__":
    p = "./"
    class FakeProj:
        target_locs_dir = p+"sc_target_locations"
        ocr_tmp_dir = p+"tmp"
    class FakeQueue:
        def put(self, x):
            return
    thr = ThreadDoInferContests(FakeQueue(), 0, FakeProj())
    print thr
    thr.run()
    exit(0)
"""
    
if __name__ == "__main__":
    paths = eval(open("../orangedata_paths").read())
    lookup = dict((x,i) for i,x in enumerate(paths))
    languages = eval(open("../orangedata_lang").read())
    languages = dict((lookup[k],v) for k,v in languages.items())
    
    equ_class(merge_contests(*pickle.load(open("../orangedata"))), languages)

    from labelcontest import LabelContest
    p = "../projects/label_grouping/"
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
    print type(internal)
    print final_grouping(internal, targets)
