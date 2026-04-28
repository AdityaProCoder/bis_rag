import json
standards = json.load(open('data/sp21_standards.json', encoding='utf-8'))
print('Total standards:', len(standards))

# Find IS 1489 entries
for s in standards:
    if '1489' in s['id']:
        print(f'IS 1489 entry: id={s["id"]!r} title={s.get("title","")[:60]!r}')

print()
# Find IS 2185 entries
for s in standards:
    if '2185' in s['id']:
        print(f'IS 2185 entry: id={s["id"]!r} title={s.get("title","")[:60]!r}')
