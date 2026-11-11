"""Propagate the external fast clock only to the physical matrix engine."""
import re
import hdl_parser as dual
from hdl_parser import require

LEAF="c209_five_mac_rtl"

def propagate(texts,top):
    definitions={}
    for file,text in texts.items():
        for name,d in dual.modules(text).items():
            require(name not in definitions,'Duplicate exported module');definitions[name]=(file,d)
    require(top in definitions and LEAF in definitions,'Missing native top/five-shard broker')
    graph={}
    for name,(file,d) in definitions.items():
        present=set(re.findall(r'\w+',dual.masked(texts[file][d['body_start']:d['end']])))
        graph[name]=dual.instances(texts[file],d,set(definitions)&present)
    reachable=set()
    def visit(name):
        if name in reachable:return
        reachable.add(name)
        require(not re.search(r'island_[ab]|c209_shared_mac_rtl|c209_attention_[qk]_pump|c209_mac[01]_rtl_service',name),'Legacy control/MAC hierarchy is reachable from new top')
        for child in graph[name]:visit(child['module'])
    visit(top)
    marked={LEAF}
    while True:
        more=marked|{n for n,children in graph.items()if any(c['module']in marked for c in children)}
        if more==marked:break
        marked=more
    paths=[]
    def walk(name,path,ancestors=()):
        require(name not in ancestors,'Recursive hierarchy')
        if name==LEAF:paths.append(path);return
        for child in graph[name]:
            if child['module']in marked:walk(child['module'],path+'/'+child['instance'],ancestors+(name,))
    walk(top,top);require(len(paths)==1,'Exactly one reachable physical broker required')
    require('/island_a'not in paths[0]and'/island_b'not in paths[0],'Legacy A/B hierarchy is not a five-shard production export')
    edits={};detail=[]
    for name in sorted(marked):
        file,d=definitions[name];text=texts[file]
        if name==LEAF:
            require(re.search(r'input\s+wire\s+mac_fast_clk\s*,\s*mac_fast_rst',text),'Verified broker fast ports missing');continue
        require(not set(dual.FAST_PORTS)&set(dual.port_declarations(text,d)),'Ancestor already clock-patched')
        changes=[(d['closing'],',\n    mac_fast_clk,\n    mac_fast_rst\n'),(d['body_start'],'\ninput mac_fast_clk;\ninput mac_fast_rst;\n')]
        linked=[]
        for child in graph[name]:
            if child['module']not in marked:continue
            body=dual.masked(text[child['opening']+1:child['closing']])
            require(re.search(r'\.\w+\s*\(',body)and not any(re.search(r'\.'+p+r'\b',body)for p in dual.FAST_PORTS),'Unknown/prepatched instance clock interface')
            changes.append((child['closing'],',\n    .mac_fast_clk(mac_fast_clk),\n    .mac_fast_rst(mac_fast_rst)\n'));linked.append(child['instance'])
        edits.setdefault(file,[]).extend(changes);detail.append(dict(module=name,file=file,connected_instances=linked,added_ports=list(dual.FAST_PORTS)))
    result=dict(texts)
    for file,changes in edits.items():
        for position,value in sorted(changes,reverse=True):result[file]=result[file][:position]+value+result[file][position:]
    return result,dict(top=top,leaf_paths=[dict(module=LEAF,path=paths[0])],ancestors=detail)
