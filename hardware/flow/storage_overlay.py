"""Map the generated U, probability and paired Q storage onto the node RAM layout."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NS='{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}'

def require(ok,message):
    if not ok: raise ValueError(message)

def write_rtl(path,body):
    with Path(path).open('w',encoding='utf-8',newline='\n') as stream:stream.write(body)

def parameter(body,name,value):
    require(re.search(r'parameter\s+(?:integer\s+)?'+name+r'\s*=\s*'+str(value)+r'\s*[,;\n]',body),
            'Generated RAM geometry differs: '+name+'='+str(value))

def apply(ip,source):
    ip,source=Path(ip),Path(source)
    folder=ip/'hdl/verilog';registrations=[]
    choices=[
        ('*update_five_to*_*u_RAM_2P_BRAM_1R1W.v','c209_attention_u_storage.v',dict(DataWidth=32,AddressWidth=13,AddressRange=5120)),
        ('*_probability_RAM_AUTO_1R1W_memcore.v','c209_attention_probability_storage.v',dict(DataWidth=256,AddressWidth=12,AddressRange=2112)),
    ]
    # Integrated score storage contains final half scores only. The old X2
    # 768-bit raw-score/header implementation is not interface-compatible.
    score_matches=list(folder.glob('*_scores_RAM_AUTO_1R1W_memcore.v'))
    require(len(score_matches)==1,'Exactly one native compact half-score RAM required')
    half_score=score_matches[0];half_body=half_score.read_text(encoding='utf-8')
    for name,value in dict(DataWidth=256,AddressWidth=9,AddressRange=512).items():
        parameter(half_body,name,value)
    for pattern,rtl,geometry in choices:
        matches=list(folder.glob(pattern));require(len(matches)==1,'One current generated RAM required: '+pattern)
        target=matches[0];original=target.read_text(encoding='utf-8')
        for name,value in geometry.items():parameter(original,name,value)
        replacement=source/'rtl'/rtl;body=replacement.read_text(encoding='utf-8')
        module=re.search(r'\bmodule\s+(\w+)',body).group(1)
        body=re.sub(r'\bmodule\s+'+re.escape(module)+r'\b','module '+target.stem,body,count=1)
        write_rtl(target,body)
        rel=target.relative_to(ip).as_posix();registrations.append(rel)
    # Q's two logical slots share a dense physical per-head bank. Port
    # expressions are taken from THIS export, never from old HLS fu numbers.
    seqs=[]
    pattern=re.compile(r'(cosmos2b_tile_\w*sequence\w*_q_RAM_2P_BRAM_1R1W)\s*#\(.*?\)\s*(q_U|q_1_U)\s*\((.*?)\n\);',re.S)
    for candidate in folder.glob('*sequence*.v'):
        body=candidate.read_text(encoding='utf-8');matches=list(pattern.finditer(body))
        if matches:seqs.append((candidate,body,matches))
    require(len(seqs)==1,'Exactly one current Q owner module required')
    sequence,body,matches=seqs[0]
    require(len(matches)==2 and [m[2] for m in matches]==['q_U','q_1_U'],'Current Q two-slot instance identity changed')
    ports=[]
    for m in matches:
        for name,value in [('DataWidth',256),('AddressRange',1280),('AddressWidth',11)]:
            require(re.search(r'\.'+name+r'\(\s*'+str(value)+r'\s*\)',m[0]),'Q geometry differs: '+name)
        values=dict(re.findall(r'\.(\w+)\(([^()]*)\)',m[3]))
        require(set(values)=={'clk','reset','address0','ce0','q0','address1','ce1','we1','d1'},'Q port contract changed')
        ports.append(values)
    require(matches[0][1]==matches[1][1],'Q slots use different memory types')
    require(not body[matches[0].end():matches[1].start()].strip(),'Logic separates Q instances')
    require(all(ports[0][k]==ports[1][k] for k in ['clk','reset','address0','address1','d1']),
            'Q slot common-address/data contract changed')
    connection=['c209_attention_q_pair_storage q_pair_storage_U (',
                ' .clk('+ports[0]['clk']+'), .reset('+ports[0]['reset']+'),']
    fields=[]
    for slot,values in enumerate(ports):
        for name in ['address0','ce0','q0','address1','ce1','we1','d1']:
            fields.append(' .q'+str(slot)+'_'+name+'('+values[name]+')')
    connection.append(',\n'.join(fields));connection.append(');')
    write_rtl(sequence,body[:matches[0].start()]+'\n'.join(connection)+body[matches[-1].end():])
    qrel='hdl/verilog/c209_attention_q_pair_storage.v';qtarget=ip/qrel
    require(not qtarget.exists(),'Q storage already patched')
    qtarget.write_bytes((source/'rtl/c209_attention_q_pair_storage.v').read_bytes())
    component=ip/'component.xml';tree=ET.parse(component);root=tree.getroot();count=0
    old_q='hdl/verilog/'+matches[0][1]+'.v'
    for fileset in root.findall(NS+'fileSets/'+NS+'fileSet'):
        for entry in list(fileset.findall(NS+'file')):
            name=(entry.findtext(NS+'name') or '').replace('\\','/')
            if name in registrations:
                kinds=entry.findall(NS+'fileType');require(len(kinds)==1,'Ambiguous storage file type')
                kinds[0].text='systemVerilogSource'
            if name==old_q:
                new=ET.Element(NS+'file');ET.SubElement(new,NS+'name').text=qrel
                ET.SubElement(new,NS+'fileType').text='systemVerilogSource'
                fileset.insert(list(fileset).index(entry)+1,new);count+=1
    require(count>=1,'Q storage did not enter the exported synthesis files')
    ET.register_namespace('spirit',NS[1:-1]);tree.write(component,encoding='utf-8',xml_declaration=True)
