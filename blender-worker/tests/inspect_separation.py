"""Read-only ownership/cut-boundary probe on an existing skinned GLB."""
import sys, json
from pathlib import Path
import bpy, bmesh, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attachment_diagnostics import branch

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=sys.argv[sys.argv.index("--")+1])
arm=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
for obj in [o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" for m in o.modifiers)]:
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
    bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table()
    deform=bm.verts.layers.deform.active
    groups={g.index:branch(g.name) for g in obj.vertex_groups}
    labels={}
    for face in bm.faces:
        totals={}
        for vert in face.verts:
            for g,w in vert[deform].items(): totals[groups[g]]=totals.get(groups[g],0)+w/len(face.verts)
        name=max(totals,key=totals.get)
        labels[face]=name if name.startswith('arm_') else 'body'
    for name in ['arm_L','arm_R']:
        pending={f for f in bm.faces if labels[f]==name};components=[]
        while pending:
            seed=pending.pop();part=[seed];todo=[seed]
            while todo:
                f=todo.pop()
                for e in f.edges:
                    for n in e.link_faces:
                        if n in pending: pending.remove(n);todo.append(n);part.append(n)
            p=np.array([tuple(obj.matrix_world@v.co) for f in part for v in f.verts])
            components.append({'faces':len(part),'bounds':[p.min(0).tolist(),p.max(0).tolist()]})
        edges={e for e in bm.edges if len(e.link_faces)==2 and sum(labels[f]==name for f in e.link_faces)==1}
        lengths=[]
        while edges:
            seed=edges.pop(); stack=[seed]; component=[seed]
            while stack:
                e=stack.pop()
                for v in e.verts:
                    for other in v.link_edges:
                        if other in edges: edges.remove(other);component.append(other);stack.append(other)
            verts={v for e in component for v in e.verts};points=np.array([tuple(obj.matrix_world@v.co) for v in verts])
            degrees=[sum(e in component for e in v.link_edges) for v in verts]
            lengths.append({'edges':len(component),'simple':all(d==2 for d in degrees),'bounds':[points.min(0).tolist(),points.max(0).tolist()]})
        print(json.dumps({'object':obj.name,'part':name,'faces':sum(n==name for n in labels.values()),'boundaries':lengths,'components':components}))
    bm.free()
