"""Add the fast clock/reset to the HLS IP-XACT interface and select SV for the MAC core."""
from pathlib import Path
import xml.etree.ElementTree as ET
import hdl_parser as dual
from hdl_parser import require

SYSTEMVERILOG_RTL=("hdl/verilog/c209_five_mac_core.v",)

def rtl_file_types(root,ip):
    ns='{'+dual.S+'}';ip=Path(ip).resolve();changes=[]
    sets=root.findall(ns+'fileSets/'+ns+'fileSet')
    for name in SYSTEMVERILOG_RTL:
        target=(ip/name).resolve();matches=[]
        for fileset in sets:
            for row in fileset.findall(ns+'file'):
                entry=(row.findtext(ns+'name') or '').replace('\\','/')
                if (ip/entry).resolve()!=target:continue
                types=row.findall(ns+'fileType')
                require(len(types)==1 and types[0].text in ('verilogSource','systemVerilogSource'),
                        'Unexpected file type for SystemVerilog dependency: '+name)
                matches.append((fileset,row,types[0]))
        require(matches,'Unregistered SystemVerilog dependency: '+name)
        for fileset,row,kind in matches:
            changes.append(dict(file=name,file_set=fileset.findtext(ns+'name'),
                                previous_type=kind.text,file_type='systemVerilogSource'))
            kind.text='systemVerilogSource'
    return changes

def metadata(path,top,version):
    tree=ET.parse(path);root=tree.getroot();ns='{'+dual.S+'}'
    require(tuple(root.findtext(ns+k) for k in ('vendor','library','name','version'))==('xilinx.com','hls',top,version),'Unexpected real native IP identity')
    ports=root.find(ns+'model/'+ns+'ports');buses=root.find(ns+'busInterfaces')
    require(ports is not None and buses is not None,'Incomplete actual IP model')
    present={p.findtext(ns+'name') for p in ports}
    require('ap_clk' in present and not set(dual.FAST_PORTS)&present,'Clock ports already transformed')
    for name in dual.FAST_PORTS:
        p=dual.sub(ports,'port');dual.sub(p,'name',name);dual.sub(dual.sub(p,'wire'),'direction','in')
    for name,kind,logical in [('mac_fast_clk','clock','CLK'),('mac_fast_rst','reset','RST')]:
        bus=dual.sub(buses,'busInterface');dual.sub(bus,'name',name)
        attrs={ns+'vendor':'xilinx.com',ns+'library':'signal',ns+'name':kind,ns+'version':'1.0'}
        dual.sub(bus,'busType',**attrs);dual.sub(bus,'abstractionType',**dict(attrs,**{ns+'name':kind+'_rtl'}));dual.sub(bus,'slave')
        mapping=dual.sub(dual.sub(bus,'portMaps'),'portMap')
        dual.sub(dual.sub(mapping,'logicalPort'),'name',logical);dual.sub(dual.sub(mapping,'physicalPort'),'name',name)
        params=dual.sub(bus,'parameters')
        for key,value in ([('FREQ_HZ','300000000'),('ASSOCIATED_RESET','mac_fast_rst')] if kind=='clock' else [('POLARITY','ACTIVE_HIGH')]):
            p=dual.sub(params,'parameter');dual.sub(p,'name',key);dual.sub(p,'value',value)
    file_types=rtl_file_types(root,Path(path).parent)
    ET.register_namespace('spirit',dual.S);tree.write(path,encoding='utf-8',xml_declaration=True)
    return file_types


