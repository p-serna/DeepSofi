/export/home1/users/bssn/serna

import os
os.chdir("/export/home1/users/bssn/serna/Olivier")
from numpy import *
def sArea():
    mesh=bmesh.from_edit_mesh(bpy.context.object.data)
    selected_verts = [v for v in mesh.verts if v.select]
    mc = array([0.,0.,0.])
    for q in range(len(selected_verts)):
        v1 = array([vs for vs in selected_verts[q].co])
        mc += v1
    mc = mc/len(selected_verts)
    vr2=0.0
    vr = 0.0
    vp = 0.0
    for q in range(len(selected_verts)):
        p = (q-1)
        if p<0: p=len(selected_verts)-1
        v1 = array([vs for vs in selected_verts[q].co])
        v2 = array([vs for vs in selected_verts[p].co])
        base = sqrt(sum((v2-v1)**2))
        vp += base
        pm = (v1+v2)/2.0
        altura = sqrt(sum((mc-pm)**2)) # altura
        vt = base*altura/2
        vr2 = vr2+vt
        #Heron formula
        Ha = sqrt(sum((v2-v1)**2))
        Hb = sqrt(sum((mc-v2)**2))
        Hc = sqrt(sum((v1-mc)**2))
        Hs = (Ha+Hb+Hc)/2.0
        vr = vr+sqrt(Hs*(Hs-Ha)*(Hs-Hb)*(Hs-Hc))
        #print([vr,vr2])
    rad = vp/2.0/pi;
    return (vr,vr2,vp,pi*rad**2,mc.tolist())



def bisects(path,nv=None,np=10):
    obj = bpy.context.object
    mesh = obj.data # Assumed that obj.type == 'MESH'
    obj.update_from_editmode() # Loads edit-mode data into object data
    selected_vertices = [v for v in mesh.vertices if v.select]
    if len(selected_vertices)>0:
        bpy.ops.mesh.select_all()
    pathd = path[1:,:]-path[:-1,]
    # number of data points
    a=[]
    np = path.shape[0]-1
    for i in range(np):
        bpy.ops.mesh.select_all()
        mc = (path[i,:]+path[i+1,:])/2.0
        nv = pathd[i,:]/sqrt(sum(pathd[i,:]**2))
        bpy.ops.mesh.bisect(plane_co=mc,plane_no=nv)
        a.append(sArea())
        bpy.ops.mesh.select_all()
    return a

def mc():
    mesh=bmesh.from_edit_mesh(bpy.context.object.data)
    selected_verts = [v for v in mesh.verts if v.select]
    mc = array([0.,0.,0.])
    for q in range(len(selected_verts)):
        v1 = array([vs for vs in selected_verts[q].co])
        mc += v1
    mc = mc/len(selected_verts)
    return(mc)
    
def get_path(mc0,mc1,nv=None,np=10):
    obj = bpy.context.object
    mesh = obj.data # Assumed that obj.type == 'MESH'
    obj.update_from_editmode() # Loads edit-mode data into object data
    selected_vertices = [v for v in mesh.vertices if v.select]
    if len(selected_vertices)>0:
        bpy.ops.mesh.select_all()
    nvt = mc2-mc1
    if nv==None:
        nv = nvt
        nv = nv/sqrt(sum((nv**2)))
    # number of data points
    dx = nvt/np
    a=[]
    for i in range(np+1):
        bpy.ops.mesh.select_all()
        bpy.ops.mesh.bisect(plane_co=mc1+dx*i,plane_no=nv)
        a.append(mc())
        bpy.ops.mesh.select_all()
    return a

path = loadtxt("test_pathT.txt")
obj = bpy.context.object
mesh = obj.data # Assumed that obj.type == 'MESH'
obj.update_from_editmode() # Loads edit-mode data into object data
bpy.ops.mesh.select_all()

pathd = path[1:,:]-path[:-1,]

for np in path:
    
