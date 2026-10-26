import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='Structural Intelligence Lab', page_icon='🏗️', layout='wide')

# Linear Euler-Bernoulli beam FEM: fixed at left, downward point load at tip.
def solve_beam(F, E, thickness, width, L, n_elem=24):
    n = n_elem + 1
    le = L / n_elem
    I = width * thickness**3 / 12.0
    EI = E * I
    K = np.zeros((2*n, 2*n))
    ke = (EI / le**3) * np.array([
        [12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]
    ], dtype=float)
    for e in range(n_elem):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(dofs, dofs)] += ke
    rhs = np.zeros(2*n)
    rhs[-2] = -F
    fixed = [0, 1]
    free = np.setdiff1d(np.arange(2*n), fixed)
    d = np.zeros(2*n)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], rhs[free])
    return np.linspace(0, L, n), d[0::2]

@st.cache_resource(show_spinner='Training POD + ML model on FEM snapshots (first run only)...')
def train_rom():
    rng = np.random.default_rng(7)
    n_elem, n_samples = 24, 320
    Fv = rng.uniform(2, 30, n_samples)
    Ev = rng.uniform(180, 410, n_samples) # GPa; spans steel, iron, and tungsten
    tv = rng.uniform(8, 20, n_samples) # mm
    bv = rng.uniform(20, 50, n_samples) # mm
    Lv = rng.uniform(.70, 1.30, n_samples)
    X = np.column_stack([Fv, Ev, tv, bv, Lv])
    W = []
    for F, E, t, b, L in X:
        _, w = solve_beam(F, E*1e9, t/1000, b/1000, L, n_elem)
        W.append(w)
    W = np.asarray(W)
    order = rng.permutation(n_samples)
    ntr = int(.8*n_samples)
    tr, te = order[:ntr], order[ntr:]
    mean = W[tr].mean(axis=0)
    centered = W[tr] - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(s*s) / max(np.sum(s*s), 1e-30)
    rank = max(1, min(5, int(np.searchsorted(energy, .9999)+1)))
    modes = Vt[:rank].T
    Ctr = centered @ modes
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    target = Ctr.ravel() if rank == 1 else Ctr
    model.fit(X[tr], target)
    cp = np.asarray(model.predict(X[te]))
    if cp.ndim == 1:
        cp = cp.reshape(-1, 1)
    wp = cp @ modes.T + mean
    info = {'rank': rank, 'energy': float(energy[rank-1]),
            'rmse': float(np.sqrt(np.mean((W[te]-wp)**2))),
            'rel': float(np.linalg.norm(W[te]-wp)/max(np.linalg.norm(W[te]), 1e-15)),
            'ntrain': len(tr), 'ntest': len(te)}
    return model, mean, modes, info

def predict_rom(model, mean, modes, F, E_GPa, t_mm, b_mm, L):
    pred = np.asarray(model.predict(np.array([[F, E_GPa, t_mm, b_mm, L]], dtype=float)))
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    return (pred @ modes.T + mean)[0]

def mesh_for_beam(x, w, width, thickness, mag, alpha):
    xs, ys, zs, intensity = [], [], [], []
    for xi, wi in zip(x, w):
        dz = mag * alpha * wi
        for yy, zz in [(-width/2, -thickness/2), (width/2, -thickness/2),
                       (width/2, thickness/2), (-width/2, thickness/2)]:
            xs.append(xi); ys.append(yy); zs.append(dz+zz); intensity.append(xi / max(float(x[-1]), 1e-12))
    ii, jj, kk = [], [], []
    for n in range(len(x)-1):
        a, b = 4*n, 4*(n+1)
        for c in range(4):
            d = (c+1) % 4
            ii += [a+c, a+c]; jj += [a+d, b+d]; kk += [b+d, b+c]
    # End caps
    ii += [0, 0, 0, 4*(len(x)-1), 4*(len(x)-1)]
    jj += [1, 2, 3, 4*(len(x)-1)+1, 4*(len(x)-1)+2]
    kk += [2, 3, 1, 4*(len(x)-1)+2, 4*(len(x)-1)+3]
    return xs, ys, zs, intensity, ii, jj, kk

def make_figure(x, wf, wr, width, thickness, mag, F, E_GPa, L, animate=True):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=('Full FEM reference', 'POD + ML prediction'), horizontal_spacing=.02)
    # Mesh trace indexes are 0 and 4; frames update those two traces.
    for col, w, color in [(1, wf, '#2878B5'), (2, wr, '#E07A32')]:
        m = mesh_for_beam(x, w, width, thickness, mag, 1.0)
        fig.add_trace(go.Mesh3d(x=m[0], y=m[1], z=m[2], i=m[4], j=m[5], k=m[6],
            intensity=m[3], colorscale=[[0.0, '#00C8FF'], [0.35, '#167DFF'], [0.65, '#FFD43B'], [1.0, '#FF2020']], cmin=0, cmax=1, showscale=False, opacity=1.0, flatshading=False,
            name='Deformed beam', hovertemplate='x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.4f} m<extra></extra>'), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=x, y=np.zeros_like(x), z=np.zeros_like(x), mode='lines',
            line=dict(color='gray', width=3, dash='dash'), name='Undeformed centreline',
            showlegend=(col==1), hoverinfo='skip'), row=1, col=col)
        # Fixed wall marker
        yw = [-width*.7, width*.7, width*.7, -width*.7]
        zw = [-thickness, -thickness, thickness, thickness]
        fig.add_trace(go.Mesh3d(x=[0]*4, y=yw, z=zw, i=[0,0], j=[1,2], k=[2,3],
            color='gray', opacity=.55, name='Fixed support', showlegend=(col==1), hoverinfo='skip'), row=1, col=col)
        fig.add_trace(go.Cone(x=[L], y=[0], z=[thickness*.8], u=[0], v=[0], w=[-max(L*.16, .08)],
            anchor='tail', sizemode='absolute', sizeref=max(L*.12,.06),
            colorscale=[[0,'#D62728'],[1,'#D62728']], showscale=False,
            name='Applied downward force', showlegend=(col==1), hovertemplate=f'Force = {F:.1f} N<extra></extra>'), row=1, col=col)
    if animate:
        frames = []
        for k, alpha in enumerate(np.linspace(0, 1, 26)):
            fdata = []
            for w, color, show in [(wf, '#2878B5', False), (wr, '#E07A32', True)]:
                m = mesh_for_beam(x, w, width, thickness, mag, alpha)
                fdata.append(go.Mesh3d(x=m[0], y=m[1], z=m[2], i=m[4], j=m[5], k=m[6],
                    intensity=m[3], colorscale=[[0.0, '#00C8FF'], [0.35, '#167DFF'], [0.65, '#FFD43B'], [1.0, '#FF2020']], cmin=0, cmax=1, showscale=False, opacity=1.0, flatshading=False))
            frames.append(go.Frame(name=str(k), data=fdata, traces=[0,4]))
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, x=.02, y=1.14,
            buttons=[dict(label='▶ Animate', method='animate', args=[None, {'frame':{'duration':45,'redraw':True}, 'transition':{'duration':0}, 'fromcurrent':True, 'mode':'immediate'}]),
                     dict(label='⏸ Pause', method='animate', args=[[None], {'frame':{'duration':0,'redraw':False}, 'transition':{'duration':0}, 'mode':'immediate'}])])])
    zspan = max(np.max(np.abs(wf)), np.max(np.abs(wr)), thickness) * mag
    for scene in ['scene','scene2']:
        fig.update_layout(**{scene: dict(xaxis_title='Length x (m)', yaxis_title='Width y (m)', zaxis_title='Vertical z (m)',
            bgcolor='#050505', aspectmode='manual', aspectratio=dict(x=2.4,y=.7,z=.9),
            xaxis=dict(range=[-.1*L,1.1*L], color='white', gridcolor='#333333', zerolinecolor='#555555', backgroundcolor='#050505'),
            yaxis=dict(range=[-width*1.4,width*1.4], color='white', gridcolor='#333333', zerolinecolor='#555555', backgroundcolor='#050505'),
            zaxis=dict(range=[-zspan*1.4-thickness,zspan*.8+thickness], color='white', gridcolor='#333333', zerolinecolor='#555555', backgroundcolor='#050505'),
            camera=dict(eye=dict(x=1.6,y=1.6,z=1.15)))})
    fig.update_layout(height=610, margin=dict(l=0,r=0,t=85,b=0), paper_bgcolor='#050505', plot_bgcolor='#050505', font=dict(color='white'),
        title=dict(text=f'3D animated cantilever | F={F:.1f} N, E={E_GPa:.0f} GPa | visual magnification {mag:.0f}×', font=dict(color='white')),
        legend=dict(orientation='h', y=-.08, font=dict(color='white'))) 
    return fig

st.markdown("<style> .stApp { background: #050505; color: #f5f5f5; } [data-testid='stSidebar'] { background: #101010; } </style>", unsafe_allow_html=True)
st.title('🏗️ Structural Intelligence Lab')
st.subheader('Interactive 3D beam • Full FEM vs POD + ML')
st.write('Change beam properties and force, rotate the 3D view, and animate the computed deformation. **The beam is rendered in 3D from a beam-element FEM solution; this is not a full 3D solid-element FEM solver. Colour runs from cool blue near the fixed support to yellow and then red at the free tip.**')
with st.sidebar:
    st.header('Beam controls')
    force = st.slider('Downward tip force (N)', 2.0, 30.0, 12.0, 1.0)
    material = st.selectbox('Beam material', ['Steel', 'Iron', 'Tungsten'], index=0)
    material_moduli = {'Steel': 200.0, 'Iron': 211.0, 'Tungsten': 400.0}
    young_gpa = material_moduli[material]
    st.caption(f"Representative Young's modulus: {young_gpa:.0f} GPa")
    length = st.slider('Beam length (m)', .70, 1.30, 1.00, .05)
    width_mm = st.slider('Beam width (mm)', 20.0, 50.0, 30.0, 1.0)
    thickness_mm = st.slider('Beam thickness (mm)', 8.0, 20.0, 12.0, .5)
    magnification = st.slider('Visual deformation magnification (×)', 1.0, 30.0, 8.0, 1.0)
    animate = st.checkbox('Enable Play/Pause animation', value=True)
    st.caption('Magnification changes only the displayed shape, not numerical displacement.')

model, mean_shape, modes, info = train_rom()
E = young_gpa*1e9; width = width_mm/1000; thickness = thickness_mm/1000
x, wf = solve_beam(force, E, thickness, width, length)
wr = predict_rom(model, mean_shape, modes, force, young_gpa, thickness_mm, width_mm, length)
err = wf-wr
rmse_case = float(np.sqrt(np.mean(err**2)))
rel_case = float(np.linalg.norm(err)/max(np.linalg.norm(wf),1e-15))
# Simple measured single-call timings; results vary with hardware and caching.
t0=time.perf_counter(); solve_beam(force,E,thickness,width,length); fem_ms=(time.perf_counter()-t0)*1000
t0=time.perf_counter(); predict_rom(model,mean_shape,modes,force,young_gpa,thickness_mm,width_mm,length); ml_ms=(time.perf_counter()-t0)*1000
cols=st.columns(4)
cols[0].metric('FEM tip displacement', f'{abs(wf[-1])*1000:.3f} mm')
cols[1].metric('ROM-ML tip displacement', f'{abs(wr[-1])*1000:.3f} mm')
cols[2].metric('Relative displacement error', f'{rel_case*100:.2f}%')
cols[3].metric('Single-call runtime', f'FEM {fem_ms:.1f} ms | ML {ml_ms:.1f} ms')
if force < 2 or force > 30 or young_gpa < 25 or young_gpa > 410:
    st.warning('Input outside model training range; interpret ML prediction cautiously.')
else:
    st.info('The input is within the ROM training range. This does not guarantee a low error; inspect the comparison.')
fig=make_figure(x,wf,wr,width,thickness,magnification,force,young_gpa,length,animate)
st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
left,right=st.columns(2)
with left:
    st.markdown('### Displacement comparison')
    f=go.Figure()
    f.add_trace(go.Scatter(x=x,y=wf*1000,mode='lines+markers',name='Full FEM'))
    f.add_trace(go.Scatter(x=x,y=wr*1000,mode='lines',line=dict(dash='dash'),name='POD + ML'))
    f.update_layout(xaxis_title='Position along beam (m)',yaxis_title='Vertical displacement (mm)',height=350, paper_bgcolor='#050505', plot_bgcolor='#050505', font=dict(color='white'), xaxis=dict(gridcolor='#333333', zerolinecolor='#555555'), yaxis=dict(gridcolor='#333333', zerolinecolor='#555555'))
    st.plotly_chart(f,use_container_width=True)
with right:
    st.markdown('### Local prediction error')
    f=go.Figure(go.Scatter(x=x,y=np.abs(err)*1000,mode='lines+markers',name='Absolute error'))
    f.update_layout(xaxis_title='Position along beam (m)',yaxis_title='Absolute error (mm)',height=350, paper_bgcolor='#050505', plot_bgcolor='#050505', font=dict(color='white'), xaxis=dict(gridcolor='#333333', zerolinecolor='#555555'), yaxis=dict(gridcolor='#333333', zerolinecolor='#555555'))
    st.plotly_chart(f,use_container_width=True)
with st.expander('Model assumptions and validation'):
    st.markdown(f"""
- **FEM:** linear Euler–Bernoulli beam elements, fixed left end, downward tip load.
- **3D view:** 1D beam FEM displacements rendered as a rectangular 3D body; not a full 3D continuum stress solver.
- **POD:** retained {info['rank']} mode(s), capturing {info['energy']*100:.4f}% of training snapshot energy.
- **Training/test cases:** {info['ntrain']} / {info['ntest']} randomly generated cases; POD basis uses training cases only.
- **Held-out test relative error:** {info['rel']*100:.3f}%.
- **Held-out test RMSE:** {info['rmse']:.3e} m.
- Runtime values are single-call timings and may vary. Use repeated runs for fair speed comparisons.
- Model assumptions: small deflection and linear elasticity. No buckling, plasticity, damage, contact, vibration, or full 3D stress field.
""")
st.caption('For your report/PPT, record actual outputs from your run. The deformation is visually magnified; displayed displacement metrics remain in physical units.')
