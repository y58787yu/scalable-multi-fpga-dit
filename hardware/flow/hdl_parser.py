"""Locate generated Verilog modules, instances and clock port declarations."""
import re
import xml.etree.ElementTree as ET

S="http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"
FAST_PORTS=("mac_fast_clk", "mac_fast_rst")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def masked(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                  lambda m: ''.join('\n' if c == '\n' else ' ' for c in m.group()), text, flags=re.S)

def close_paren(text, opening):
    require(text[opening] == '(', 'Expected opening parenthesis')
    depth = 0
    for i in range(opening, len(text)):
        depth += int(text[i] == '(') - int(text[i] == ')')
        if depth == 0:
            return i
    raise ValueError('Unterminated Verilog parenthesis')

def modules(text):
    clean = masked(text)
    result = {}
    for match in re.finditer(r'\bmodule\s+(\w+)\b', clean):
        end = clean.find('endmodule', match.end())
        require(end >= 0, 'Unterminated module: ' + match[1])
        opening = clean.find('(', match.end())
        cursor = match.end()
        while clean[cursor].isspace():
            cursor += 1
        if clean[cursor] == '#':
            opening = clean.find('(', cursor)
            cursor = close_paren(clean, opening) + 1
            opening = clean.find('(', cursor)
        closing = close_paren(clean, opening)
        semicolon = closing+1
        while clean[semicolon].isspace():
            semicolon += 1
        require(opening < closing < semicolon < end and clean[semicolon] == ';', 'Unsupported module header')
        require(match[1] not in result, 'Duplicate module definition')
        result[match[1]] = dict(start=match.start(), end=end+len('endmodule'),
                              opening=opening, closing=closing, body_start=semicolon+1)
    return result

def instances(text, definition, names):
    clean = masked(text)
    result = []
    for name in names:
        pattern = r'\b' + re.escape(name) + r'\b\s*(?:#\s*\()?' 
        for match in re.finditer(pattern, clean[definition['body_start']:definition['end']]):
            cursor = definition['body_start'] + match.end()
            if clean[cursor-1] == '(':
                cursor = close_paren(clean, cursor-1) + 1
            remaining = clean[cursor:definition['end']]
            inst = re.match(r'\s*(\w+)\s*\(', remaining)
            if not inst:
                continue
            opening = cursor + inst.end() - 1
            closing = close_paren(clean, opening)
            require(clean[closing+1:].lstrip().startswith(';'), 'Unsupported instance form')
            result.append(dict(module=name, instance=inst[1], opening=opening, closing=closing))
    return result

def port_declarations(text, definition):
    header = text[definition['opening']+1:definition['closing']]
    require(not re.search(r'\b(input|output|inout)\b', masked(header)),
            'Port propagator intentionally accepts only native HLS non-ANSI ancestor headers')
    return [p.strip() for p in header.split(',') if p.strip()]

def sub(node, name, text=None, **attrs):
    item = ET.SubElement(node, '{'+S+'}'+name, attrs)
    item.text = text
    return item

