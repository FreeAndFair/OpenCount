from PIL import Image, ImageDraw
import os
from random import random
from collections import Counter

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

def find_lines(data):
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
            if True:
                #print 'black at', y,x
                
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
                        lines.append(("H", (l,y-3,r+10,y+3)))
            """
            except Exception as e:
                print e
                pass
            """
    
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
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None
def intersect1(line1, line2):
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None
def union(line1, line2):
    top = min(line1[1], line2[1])
    bottom = max(line1[3], line2[3])
    left = min(line1[0], line2[0])
    right = max(line1[2], line2[2])
    return left, top, right, bottom

def dfs(graph, start):
    s = [start]
    seen = {}
    while s != []:
        top = s.pop()
        if top in seen: continue
        seen[top] = True
        s += graph[top]
    return seen.keys()

def to_graph(lines, width, height):
    print width, height
    table = [[None]*(width+20) for _ in range(height+20)]
    equal = []
    for full in lines:
        if full[0] != 'H': continue
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

        
    vertexes = dict((x, []) for _,x in lines)

    boxes = []
    for way1,line1 in lines:
        for way2,line2 in lines:
            if way1 != way2:
                if intersect(line1, line2):
                    boxes.append(intersect(line1, line2))
                    vertexes[line1].append(line2)
    return boxes,dict((k,v) for k,v in vertexes.items() if v != [])

def find_squares(graph):
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
    def area(x): 
        if x == None: return 0
        return (x[2]-x[0])*(x[3]-x[1])

    targets = [x for x in giventargets]
    avg_targ_area = sum(map(area,targets))/len(targets)
    squares = [x for x in squares if area(x) > avg_targ_area*2]
    """
    ## Begin hack since I have to deal with my own targets
    bigest_box = max(squares, key=area)
    squares = [x for x in squares if intersect(x, bigest_box) == x]
    print len(squares), 'remain'
    ## End hack
    
    targets = [x for x in squares if area(x) < 10000 and 20 < x[3]-x[1] < 80 and 40 < x[2]-x[0] < 80]
    
    lgraph = [(x,y) for x in targets for y in targets if intersect(x,y) != None]
    #print lgraph
    graph = reduce(lambda p,c: p[c[0]].append(c[1]) or p, lgraph, dict((x,[]) for x in targets))
    #print graph
    #print 'before', targets
    targets = []
    while graph != {}:
        tounion = dfs(graph, graph.keys()[0])
        targets.append(reduce(union,tounion))
        for k in tounion:
            del graph[k]
    #print 'after', targets
    """
    contests = []

    #print "T", targets
    for sq in sorted(squares, key=area):
        if sq in targets: continue
        inside = [t for t in targets if area(intersect(sq, t)) == area(t)]
        if inside != []:
            # Check if there are other targets inside of this contest bounding box.
            otherinside = [t for t in giventargets if t not in inside and area(intersect(sq, t)) == area(t)]
            if otherinside != []:
                print "HAD AN ERROR ON", name
            contests.append(sq)
            targets = [x for x in targets if x not in inside]

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
        for box in giventargets:
            imd.rectangle(box, fill=(255,0,0))
        new.save(tmp+"/qqq.png")

    return contests

    #print targets, contests
    #os.popen("open tmp/*")
    #exit(0)
        

def extract_contest(image_path, giventargets):
    #Image.open(image_path).save(tmp+"/"+image_path.split("/")[-1][:-4]+"-orig.png")
    data = load_num(image_path)
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

    return data, do_extract(image_path.split("/")[-1], 
                            load_pil(image_path), squares, giventargets)

def ballot_preprocess(i, f, image, contests, targets, lang):
    print 'making', f, f.split("/")[-1].split(".")[0]
    sub = os.path.join(tmp+"", f.split("/")[-1].split(".")[0]+"-dir")
    os.mkdir(sub)
    res = []
    for c in contests:
        os.mkdir(os.path.join(sub, "-".join(map(str,c))))
        t = compare_preprocess(lang, os.path.join(sub, "-".join(map(str,c))), 
                               image, c, targets)
        res.append((i, c, t))
    print "RESULTING", res
    return res


def compare_preprocess(lang, path, image, contest, targets):
    #print contest, targets
    #print [intersect(contest, x) for x in targets]
    #print 'all', targets
    targets = [x for x in targets if intersect(contest, x) == x]
    l,u,r,d = contest
    cont_area = load_num(pilimg=num2pil(image).crop((l+10,u+10,r-10,d-10)))
    print contest
    #print "bottom of box", d
    #print 'targets', targets
    tops = [0]+sorted([a[1]-u-10 for a in targets])+[d]
    #print contest
    #print "USING", tops
    blocks = []
    for upper,lower in zip(tops, tops[1:]):
        print upper, lower
        img = num2pil(cont_area[upper:lower])
        name = os.path.join(path, str(upper)+".tif")
        img.save(name)
        os.popen("tesseract %s %s -l %s"%(name, name, lang))
        istarget = (upper != 0)
        if os.path.exists(name+".txt"):
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
    print 'running with1', otexts1
    print 'running with2', otexts2
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
    weights = sorted([(row_dist(a,b),a,b) for a in texts1 for b in texts2])

    print "11111"
    for each in enumerate(targtext1): print each
    print "22222"
    for each in enumerate(targtext2): print each

    val = 0
    matching = []
    while texts1 != {} and texts2 != {}:
        for weight,a,b in weights:
            if a in texts1 and b in texts2 and istargs1[a] == istargs2[b]:
                if istargs1[a] == True:
                    print 'together', targtext1.index(a), targtext2.index(b)
                    matching.append((targtext1.index(a),
                                     targtext2.index(b)))
                val += weight
                del texts1[a]
                del texts2[b]
                break
    print "MATCHING", matching
    return float(val+sum(texts1.values())+sum(texts1.values()))/size, matching

def first_pass(contests):
    ht = {}
    for each in contests:
        if len(each[2]) not in ht: ht[len(each[2])] = []
        ht[len(each[2])].append(each)
    return ht.values()

def merge_equal(contests):
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
        result += merge_equal(group)
    print "RETURNING", result
    return result

def merge_contests(ballot_data, fulltargets):
    new_data = []
    for ballot, targets in zip(ballot_data, fulltargets):
        print 'next'
        new_ballot = []
        for group in targets:
            equal = [i for t in group for i,(_,bounding,_) in enumerate(ballot) if intersect(t, bounding)]
            print 'add', list(set(equal))
            merged = sum([ballot[x][2] for x in list(set(equal))],[])
            new_ballot.append((ballot[x][0], ballot[x][1], merged))
        new_data.append(new_ballot)
    print new_data
    print new_data == ballot_data
    return new_data
            

def do_grouping(t, paths, giventargets, lang_map = {}):
    global tmp
    print "ARGUMENTS", (t, paths, giventargets, lang_map)
    print 'giventargets', giventargets
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    ballots = []
    print "LEN", len(giventargets), len(paths)
    for i,f in enumerate(paths):
        print f
        im, contests = extract_contest(f, sum(giventargets[i],[]))
        lang = lang_map[f] if f in lang_map else 'eng'
        get = ballot_preprocess(i, f, im, contests, sum(giventargets[i],[]), lang)
        ballots.append(get)
    print "WORKING ON", ballots
    ballots = merge_contests(ballots, giventargets)
    return equ_class(ballots)

#print do_grouping("tmp", ["/home/nicholas/playaround/chi_small/"+x for x in os.listdir("/home/nicholas/playaround/chi_small/")])

#equ_class(merge_contests([[(0, (1110, 281, 1600, 592), [(False, u'\n'), (True, u'Z DIANNE Fenwsrem\nParty Prelcrenoex Democraiic\nUnited States Senator\n\n'), (True, u'\xa4 c0u.EEN sum FERNALD\nParty Preference Democratic\nMctrxer/Co\u20221sn1|an\\/Artrsl\n\n'), (True, u'EI\n\n')]), (0, (1111, 2139, 1600, 2472), [(False, u'Judg\xa40|01\xa4Sn4>\xa4ri\xa4rO\xa411rt\nO|\u2018HceN0.1\nV0te|0rOne\n\n'), (True, u'E Eucmz .uz\u2022-mx\nGeneral Praciioe Atmmey\n\n'), (True, u'\xa4 \xa4Ea0RA\u2022-\u2022 .1. cv\xb7auANcs\nJudge OI me Superior Court\n\n'), (True, u'EI\n\n')]), (0, (1110, 586, 1600, 1095), [(False, u'UNITED STATES REPRESENTATIVE\n48th District\nVote lor One\n\n'), (True, u'\xa4 mm Roe-\u2022RABAcHER\nParty Prelerencex Republican\nLIS. Representative\n\n'), (True, u'\xa4 ALAN scr-num\nParty Prelerencez None\nMarkebng Sales Executive\n\n'), (True, u'D Ron vARAsTEH\nParty Preleaenoez Democratic\nEngineer/Small Businessman\n\n'), (True, u'III\n\n')]), (0, (1111, 1089, 1600, 1835), [(False, u'MEMBER OF THE STATE ASSEMBLY\n72nd District\nVote kx Ons\n\n'), (True, u'E \xb7rRAvns ALLEN\nParty Preluencez Republican\nSmall Business Owner\n\n'), (True, u'D ALBERT AvALA\nPany Prelerencec Democratic\nRetired Poiioe Commander\n\n'), (True, u'E Joe Dovmn\nParty Preference: Democratic\nCity C0mmissi0rverIBusinesspers0n\n\n'), (True, u'E LONG P1-uuva\nPaty Prelerence: Republican\nMember. Orange County Board 07\nEducation\n\n'), (True, u'\xa4 may EDGAR\nPany Preference Republican\nBusinessman/Mayor\n\n'), (True, u'EI\n\n')]), (0, (619, 282, 1116, 2650), [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'D ELIZABETH EMKEN\nParty Preierence Republican\nBusinessw0rnanINcnpr\xa4|it Executive\n\n'), (True, u'D \xbb<AanRu\xa4\xa4nN KARIM ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E Rncx w|LuAMs\nParty Prelerencer Republican\nBusiness Aticmey\n\n'), (True, u'E R0<sEu0 T. G\xa4.0R\xa4A\nPany Preierence: Republican\nGraduate Student/Businessman\n\n'), (True, u'E DON J. GRUNDMANN\nParty Preference: American lndependen\nDoctor 01 Chiropractic\n\n'), (True, u'D Roazm LAUTEN\nParty Preference; Republican\n\n'), (True, u'E ami. K. ucv-m=00r\nParty Preference: Libertarian\nRetired Nurse\n\n'), (True, u'D \xa4Avn\xa4 ALEx Lsvnrr\nParty Prelerence; Democratic\nComputer S<:ientistlEngineer\n\n'), (True, u'\xa4 0RLY mrz\nParty Preference: Republi \xbb n\nDoctor/Attorney/Businessworrran\n\n'), (True, u'E AL RAMrREz\nParty Preference; Republican\nBusinessman\n\n'), (True, u'\xa4 DIRK ALLEN \u2022<0NOPn<\nParty Preference: Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republican\nRetired Administration Director\n\n'), (True, u'Q MIKE smnmuwc;\nParty Preference; Demouatic\nConsumer Rights Attomey\n\n'), (True, u'Q cum; srewmr\nParty Preference; Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'Q NAK sHA\xbb-1\nParty Prelevenoe; Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 mcnum sunmzm\nParty Preierence: Republican\nEducator/Author/Businessrr1an\n\n'), (True, u'E DENNIS JAc\u2022<s0N\nParty Prelererxce Republican\nAerospace General Manager\n\n'), (True, u'\xa4 DAN HUGHES\nParty Pralerence: Republican\nSmall Business Owner\n\n'), (True, u'E GREG <:0r~1L0N\nParty Preference: Republican\nBusmessman/CPA\n\n'), (True, u'E JOHN B0Rur=+=\nParty Preference, Republican\nBusinessman\n\n'), (True, u'D OSCAR ALEJANDRO BRAUN\nParty Preference: Republican\nBusinessman/Rancher\n\n'), (True, u'\xa4 MARSHA Fznwumn\nParty Prelerence: Peace and Freedom\nRehred Teacher\n\n')])], [(1, (1108, 282, 1599, 595), [(False, u'\n'), (True, u'\xa4 Rosem LAUTEN\nParty Preference: Repubiican\n\n'), (True, u'D GAIL K, i.icHTFOOT\nPariy Prelerencet Libertarian\nRetired Nurse\n\n'), (True, u'E\n\n')]), (1, (1108, 2187, 1599, 2531), [(False, u'Judq\xa4\xa41t!1sSup\u2022ricrCc\u2022:i\nO||icoN\xa4.1\nV\xa4t\xa4i\xa4r0n\xa4\n\n'), (True, u'\xa4 EUGENE Jrzr-1A\xbb<\nGeneral Practice Attorney\n\n'), (True, u'Z DEBORAH .1, co-\u2022uANc\nJudge 0i me Supenov Coun\n\n'), (True, u'III\n\n')]), (1, (1108, 1090, 1599, 1492), [(False, u'STATE SENATOR\n29lh Disirld\nVob br Ona\n\n'), (True, u"D GREG DIAMOND\nParty Preierence: Democratic\nW\xa4\u20221<ers' Rights Attorney\n\n"), (True, u'\xa4 R0BERT "BOB" \u2022-uur=r=\nParty Preference: Republican\nLawmaksrlBusiness Ouncv\n\n'), (True, u"IZ'!\n\n")]), (1, (1108, 1486, 1599, 1890), [(False, u'MEMBER OF THE STATE ASSEMBLY\n5501 Disirid\nVon kx Ons\n\n'), (True, u'\xa4 cum HAGMAN\nParty Prelerenca Republican\nBusmess Owner/Assemblyman\n\n'), (True, u'E GREGG 0 \u2022=RncH|.E\nPany Prelcvemez Democratic\nScual Worker\n\n'), (True, u'Z\n\n')]), (1, (1108, 589, 1599, 1096), [(False, u'UNITED STATES REPRESENTATIVE\n3901 Dlsidd\nVcts br Ons\n\n'), (True, u"Q D'MAR|E Mummenn\nParty Pvelarence None\nCommunity Volunteer\n\n"), (True, u'E .1Av cnen\nParty Prelerems: Democratic\nBusmessman/School Bcardmembev\n\n'), (True, u'\xa4 an Rovce\nParty Prelersncez Repustican\nU.S. Representative\n\n'), (True, u'IZ\n\n')]), (1, (619, 284, 1114, 2648), [(False, u'UNITED STATES SENATOR\nVote lor Ono\n\n'), (True, u'Z \xa4Av\u2022\xa4 ALEX uzvm\nParty Preference: Democratic\nComputer Scientist/Engineer\n\n'), (True, u'\xa4 0RLv mnz\nParty Preference; Republican\nD0c|0rIAtt\xa4rr\u2022ey/Busiruesswcrnarw\n\n'), (True, u'Q AL RAMIREZ\nParty Prelecenoe; Republican\nBusinessman\n\n'), (True, u'Z uma ALLEN \u2022<0N0r>n<\nParty Prelerence; Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference; Rcpmuimn\nRedred Adninisuation Directv!\n\n'), (True, u'\xa4 MIKE smnmuucs\nParty Preference: Democratic\nConsuner Rnghts Attorney\n\n'), (True, u'E DIANE sTEwART\nParty Preference: Democrauc\nBusinessuunan/Firnarxce Manager\n\n'), (True, u'E MAK sum\nParty Preterenoe; Democrats\nEnvironmental Health Consultant\n\n'), (True, u'E wacuum sr-unasw\nParty Preference: Republiwn\nEdu\xa22tonAulr\u2022\xa4vlBusirmessrnar\xa4\n\n'), (True, u'E uemms .1Ac\u2022<s0N\nParty Preference: Republican\nAerospace GQVEFBI Manager\n\n'), (True, u'E DAN Hucv-eas\nParty Preference: Republican\nSmall Busaness Owner\n\n'), (True, u'Z GREG c0N\xa4.0~\nParty Preference; Republkzn\nBusinessman/CPA\n\n'), (True, u'E ,10HN BORUFF\nParty Preference: Republi \xbb= n\nBusinessman\n\n'), (True, u'E oscm Aumuuno smuu\nParty Preference: Republican\nBusinessman/Rznd\u2022er\n\n'), (True, u'D MARsHA FEINLAND\nParty Prelerence; Peace and Freedom\nRetired Teadmer\n\n'), (True, u'Q DIANNE Femswsou\nParty Pvelyenoez Democratic\nUnited States Senator\n\n'), (True, u'\xa4 c01.Lea~ sum FERNAL0\nParty Prelerence; Dcmocrahc\nMoliner/C0r\u2022saa|taa\xb7\xa4\u2022/Artist\n\n'), (True, u'\xa4 EuzABErH EMKEN\nParty Prelerevmz Republican\nBusines$w<\xa4nar\xa4IN0r\\g1r0Iit Executive\n\n'), (True, u'E wxnnnunnnw manu ALI\nParty Prelerence: Peace and Freedom\nBusinessman\n\n'), (True, u'\xa4 Rncx wnLuAMs\nParty Preference: Republrwn\nBusmess Attorney\n\n'), (True, u'E Rocsuo T. cn.0mA\nParty Preference: Republican\nGraduate Studen\\l\u2018Businessman\n\n'), (True, u'E DON J. GRUNDMANN\nPady Preference: American Independen\nDoctor of Chiropractic\n\n')])]], [[[(635, 366, 702, 404), (636, 469, 703, 507), (635, 573, 702, 611), (635, 678, 702, 716), (635, 781, 702, 819), (635, 886, 702, 924), (635, 989, 702, 1027), (635, 1093, 702, 1131), (635, 1196, 702, 1234), (635, 1301, 702, 1339), (637, 1403, 704, 1441), (635, 1508, 702, 1546), (636, 1611, 703, 1649), (636, 1715, 703, 1753), (636, 1819, 703, 1857), (636, 1923, 703, 1961), (636, 2027, 703, 2065), (636, 2130, 703, 2168), (636, 2234, 703, 2272), (637, 2337, 704, 2375), (636, 2442, 703, 2480), (638, 2544, 705, 2582), (1125, 304, 1192, 342), (1125, 407, 1192, 445), (1125, 511, 1192, 549)], [(1125, 701, 1192, 739), (1126, 805, 1193, 843), (1125, 911, 1192, 949), (1125, 1013, 1192, 1051)], [(1126, 1203, 1193, 1241), (1126, 1307, 1193, 1345), (1126, 1411, 1193, 1449), (1126, 1515, 1193, 1553)], [(1126, 1650, 1193, 1688), (1125, 1753, 1192, 1791)], [(1126, 2250, 1193, 2288), (1126, 2323, 1193, 2361), (1126, 2395, 1193, 2433)]], [[(635, 368, 702, 406), (636, 472, 703, 510), (634, 576, 701, 614), (635, 679, 702, 717), (635, 782, 702, 820), (634, 886, 701, 924), (634, 990, 701, 1028), (636, 1093, 703, 1131), (634, 1197, 701, 1235), (634, 1301, 701, 1339), (635, 1403, 702, 1441), (635, 1506, 702, 1544), (635, 1610, 702, 1648), (634, 1714, 701, 1752), (634, 1818, 701, 1856), (636, 1921, 703, 1959), (634, 2025, 701, 2063), (636, 2126, 703, 2164), (635, 2231, 702, 2269), (635, 2334, 702, 2372), (636, 2438, 703, 2476), (636, 2541, 703, 2579), (1124, 306, 1191, 344), (1124, 411, 1191, 449), (1124, 514, 1191, 552)], [(1124, 703, 1191, 741), (1124, 807, 1191, 845), (1123, 911, 1190, 949), (1124, 1014, 1191, 1052)], [(1123, 1205, 1190, 1243), (1124, 1308, 1191, 1346), (1124, 1412, 1191, 1450)], [(1123, 1602, 1190, 1640), (1124, 1705, 1191, 1743), (1123, 1809, 1190, 1847)], [(1124, 2303, 1191, 2341), (1123, 2376, 1190, 2414), (1124, 2449, 1191, 2487)]]]))

#do_grouping('ocr_tmp_dir', ['/home/nicholas/googlecode/opencount/opencount/projects/orange/blankballots_straight/339_3116_1_36_1.png', '/home/nicholas/googlecode/opencount/opencount/projects/orange/blankballots_straight/339_3115_1_34_1.png'], [[[(635, 366, 702, 404), (636, 469, 703, 507), (635, 573, 702, 611), (635, 678, 702, 716), (635, 781, 702, 819), (635, 886, 702, 924), (635, 989, 702, 1027), (635, 1093, 702, 1131), (635, 1196, 702, 1234), (635, 1301, 702, 1339), (637, 1403, 704, 1441), (635, 1508, 702, 1546), (636, 1611, 703, 1649), (636, 1715, 703, 1753), (636, 1819, 703, 1857), (636, 1923, 703, 1961), (636, 2027, 703, 2065), (636, 2130, 703, 2168), (636, 2234, 703, 2272), (637, 2337, 704, 2375), (636, 2442, 703, 2480), (638, 2544, 705, 2582)], [(1125, 304, 1192, 342), (1125, 407, 1192, 445), (1125, 511, 1192, 549)], [(1125, 701, 1192, 739), (1126, 805, 1193, 843), (1125, 911, 1192, 949), (1125, 1013, 1192, 1051)], [(1126, 1203, 1193, 1241), (1126, 1307, 1193, 1345), (1126, 1411, 1193, 1449), (1126, 1515, 1193, 1553)], [(1126, 1650, 1193, 1688), (1125, 1753, 1192, 1791)], [(1126, 2250, 1193, 2288), (1126, 2323, 1193, 2361), (1126, 2395, 1193, 2433)]], [[(635, 368, 702, 406), (636, 472, 703, 510), (634, 576, 701, 614), (635, 679, 702, 717), (635, 782, 702, 820), (634, 886, 701, 924), (634, 990, 701, 1028), (636, 1093, 703, 1131), (634, 1197, 701, 1235), (634, 1301, 701, 1339), (635, 1403, 702, 1441), (635, 1506, 702, 1544), (635, 1610, 702, 1648), (634, 1714, 701, 1752), (634, 1818, 701, 1856), (636, 1921, 703, 1959), (634, 2025, 701, 2063), (636, 2126, 703, 2164), (635, 2231, 702, 2269), (635, 2334, 702, 2372), (636, 2438, 703, 2476), (636, 2541, 703, 2579)], [(1124, 306, 1191, 344), (1124, 411, 1191, 449), (1124, 514, 1191, 552)], [(1124, 703, 1191, 741), (1124, 807, 1191, 845), (1123, 911, 1190, 949), (1124, 1014, 1191, 1052)], [(1123, 1205, 1190, 1243), (1124, 1308, 1191, 1346), (1124, 1412, 1191, 1450)], [(1123, 1602, 1190, 1640), (1124, 1705, 1191, 1743), (1123, 1809, 1190, 1847)], [(1124, 2303, 1191, 2341), (1123, 2376, 1190, 2414), (1124, 2449, 1191, 2487)]]], {})
