document.addEventListener('DOMContentLoaded', () => {
    const alignForm = document.getElementById('align-form');
    const beforeInput = document.getElementById('before-file');
    const afterInput = document.getElementById('after-file');
    const annInput = document.getElementById('ann-file');
    const samToggle = document.getElementById('sam-toggle');
    const btnProcess = document.getElementById('btn-process');
    const btnExportRoboflow = document.getElementById('btn-export-roboflow');

    const nameBefore = document.getElementById('name-before');
    const nameAfter = document.getElementById('name-after');
    const nameAnn = document.getElementById('name-ann');

    const mFrame = document.getElementById('m-frame');
    const mInliers = document.getElementById('m-inliers');
    const mRatio = document.getElementById('m-ratio');
    const mInstances = document.getElementById('m-instances');

    const imgSemBefore = document.getElementById('img-sem-before');
    const imgSemAfter = document.getElementById('img-sem-after');
    const imgMaskRgb = document.getElementById('img-mask-rgb');
    const imgMaskScaled = document.getElementById('img-mask-scaled');
    const imgMatches = document.getElementById('img-matches');
    const imgRawBefore = document.getElementById('img-raw-before');
    const imgRawAfter = document.getElementById('img-raw-after');

    // Setup dropzone file name updates
    beforeInput.addEventListener('change', (e) => {
        if (e.target.files.length === 1) {
            nameBefore.textContent = e.target.files[0].name;
        } else if (e.target.files.length > 1) {
            nameBefore.textContent = `${e.target.files.length} sequence frames selected`;
        }
    });

    afterInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) nameAfter.textContent = e.target.files[0].name;
    });

    annInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) nameAnn.textContent = e.target.files[0].name;
    });

    // Tab switching logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Align Form Submission
    alignForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!beforeInput.files.length || !afterInput.files[0]) {
            alert('Please select Before image(s)/frames and After image.');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < beforeInput.files.length; i++) {
            formData.append('before_files', beforeInput.files[i]);
        }
        formData.append('after_file', afterInput.files[0]);
        if (annInput.files[0]) {
            formData.append('ann_file', annInput.files[0]);
        }
        formData.append('run_sam_refine', samToggle.checked ? 'true' : 'false');

        btnProcess.disabled = true;
        btnProcess.textContent = beforeInput.files.length > 1 
            ? `Matching Best Frame across ${beforeInput.files.length} frames...` 
            : 'Processing SIFT & Masks...';

        try {
            const res = await fetch('/api/align', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Alignment failed.');
            }

            const data = await res.json();
            
            // Update metrics
            if (mFrame) mFrame.textContent = data.metrics.selected_frame_name || 'Single File';
            mInliers.textContent = data.metrics.inliers_count;
            mRatio.textContent = data.metrics.inlier_ratio;
            mInstances.textContent = data.metrics.annotations_count;

            // Update images
            imgSemBefore.src = data.images.semantic_overlay_before;
            imgSemAfter.src = data.images.semantic_overlay_after;
            if (imgMaskRgb && data.images.semantic_rgb_mask) imgMaskRgb.src = data.images.semantic_rgb_mask;
            if (imgMaskScaled && data.images.semantic_scaled_mask) imgMaskScaled.src = data.images.semantic_scaled_mask;
            imgMatches.src = data.images.matches_overlay;
            imgRawBefore.src = data.images.before_aligned;
            imgRawAfter.src = data.images.after_reference;


        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            btnProcess.disabled = false;
            btnProcess.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Align & Generate Masks`;
        }
    });

    // Roboflow Export Button
    btnExportRoboflow.addEventListener('click', async () => {
        if (!beforeInput.files[0] || !afterInput.files[0] || !annInput.files[0]) {
            alert('Please select Before image, After image, and Roboflow Annotations file to export Roboflow SAM ZIP.');
            return;
        }

        const formData = new FormData();
        formData.append('before_file', beforeInput.files[0]);
        formData.append('after_file', afterInput.files[0]);
        formData.append('ann_file', annInput.files[0]);

        btnExportRoboflow.disabled = true;
        btnExportRoboflow.textContent = 'Packaging ZIP...';

        try {
            const res = await fetch('/api/export-roboflow-sam', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Failed to create export zip.');

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'roboflow_sam_export.zip';
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            alert(`Export Error: ${err.message}`);
        } finally {
            btnExportRoboflow.disabled = false;
            btnExportRoboflow.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> Export Roboflow SAM ZIP`;
        }
    });
});
