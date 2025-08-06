$(document).ready(function() {
  const socket = io();
  let currentStateCode = null;
  let cachedStates = null;
  let trackIds = new Set();
  let clientId = null;
  let stateCounts = {};
  let stateDurations = {};
  let currentPage = 1;
  let totalPages = 0;
  let chartUpdateInterval = null;
  let lastFrameTime = 0;
  const frameTimeout = 5000; // 5 секунд таймаут для видео

  // Dark Mode Toggle
  $('#darkModeToggle').on('click', function() {
    const isDark = $('html').attr('data-theme') === 'dark';
    $('html').attr('data-theme', isDark ? 'light' : 'dark');
    $(this).text(isDark ? '🌙 Dark Mode' : '☀️ Light Mode');
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
  });

  // Восстановить тему из localStorage
  if (localStorage.getItem('theme') === 'dark') {
    $('html').attr('data-theme', 'dark');
    $('#darkModeToggle').text('☀️ Light Mode');
  }

  // Регистрация Service Worker для PWA
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/js/service-worker.js')
      .then(reg => console.log('Service Worker зарегистрирован'))
      .catch(err => console.error('Ошибка регистрации Service Worker:', err));
  }

  function resetStats() {
    stateCounts = {};
    stateDurations = {};
    trackIds.clear();
    $('#totalPigs').text('0');
    $('#activePigs').text('0%');
    $('#avgStateTime').text('0 мин');
    $('#logTable tbody').empty();
    $('#sortTrack').empty().append('<option value="">Выберите ID свиньи</option>');
    $('#chartTrackFilter').empty().append('<option value="">Все ID</option>');
    initChart(null);
    initDurationChart(null);
  }

  function initChart(trackId) {
    let data = trackId ? (stateCounts[trackId] || {}) : Object.keys(stateCounts).reduce((acc, tid) => {
      for (let state in stateCounts[tid]) {
        acc[state] = (acc[state] || 0) + stateCounts[tid][state];
      }
      return acc;
    }, {});

    const trace = {
      x: Object.keys(data),
      y: Object.values(data),
      type: 'bar',
      marker: { color: '#6e8efb' }
    };

    const layout = {
      height: 300,
      margin: { t: 20, b: 100 },
      xaxis: { title: 'Состояние', tickangle: 45 },
      yaxis: { title: 'Количество' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('stateChart', [trace], layout, { responsive: true });
  }

  function initDurationChart(trackId) {
    let data = trackId ? (stateDurations[trackId] || {}) : Object.keys(stateDurations).reduce((acc, tid) => {
      for (let state in stateDurations[tid]) {
        acc[state] = (acc[state] || 0) + stateDurations[tid][state];
      }
      return acc;
    }, {});

    const trace = {
      x: Object.keys(data),
      y: Object.values(data).map(seconds => Math.round(seconds / 60)),
      type: 'bar',
      marker: { color: '#a777e3' }
    };

    const layout = {
      height: 300,
      margin: { t: 20, b: 100 },
      xaxis: { title: 'Состояние', tickangle: 45 },
      yaxis: { title: 'Длительность (мин)' },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('durationChart', [trace], layout, { responsive: true });
  }

  function updateKPI() {
    const totalPigs = Object.keys(stateCounts).length;
    let activePigs = 0;
    let totalTime = 0;
    let stateCount = 0;

    for (let trackId in stateCounts) {
      if (stateCounts[trackId]['Feeding'] || stateCounts[trackId]['Sitting'] || stateCounts[trackId]['Standing']) {
        activePigs++;
      }
      if (stateDurations[trackId]) {
        for (let state in stateDurations[trackId]) {
          totalTime += stateDurations[trackId][state];
          stateCount++;
        }
      }
    }

    $('#totalPigs').text(totalPigs);
    $('#activePigs').text(totalPigs ? Math.round((activePigs / totalPigs) * 100) + '%' : '0%');
    $('#avgStateTime').text(stateCount ? Math.round(totalTime / stateCount / 60) + ' мин' : '0 мин');
  }

  function startChartUpdates() {
    if (chartUpdateInterval) {
      clearInterval(chartUpdateInterval);
    }
    chartUpdateInterval = setInterval(() => {
      loadChartData();
    }, 3000);
  }

  function loadInitialData() {
    $.ajax({
      url: '/get_states',
      method: 'GET',
      success: function(data) {
        cachedStates = data.states;
        loadLogs();
        loadChartData();
        startChartUpdates();
      },
      error: function(xhr) {
        showAlert('danger', 'Ошибка загрузки состояний: ' + (xhr.responseJSON?.error || 'Неизвестная ошибка'));
      }
    });
  }

  function loadLogs() {
    socket.emit('request_logs', { clientId: clientId });
  }

  function loadChartData() {
    socket.emit('request_chart_data', { clientId: clientId });
  }

  function updateLogTable(logs) {
    const filterTime = $('#filterTime').val().trim();
    const filterTrack = $('#filterTrack').val().trim();
    const sortTrack = $('#sortTrack').val();
    const tbody = $("#logTable tbody");
    tbody.empty();

    let filteredLogs = sortTrack ? logs.filter(item => String(item[0]) === sortTrack) : logs.slice(-10);

    filteredLogs.forEach(function(item) {
      const track_id = item[0];
      const message = item[1];
      trackIds.add(String(track_id));

      if (filterTrack && !String(track_id).includes(filterTrack)) return;
      if (filterTime && !message.includes(filterTime)) return;

      const timePart = message.split(" - ")[0];
      const row = `<tr>
                    <td>${timePart}</td>
                    <td>${track_id}</td>
                    <td>${message}</td>
                  </tr>`;
      tbody.append(row);
    });

    updateTrackFilterOptions();
  }

  function updateTrackFilterOptions() {
    const sortTrack = $('#sortTrack');
    const chartTrack = $('#chartTrackFilter');
    const currentSortVal = sortTrack.val();
    const currentChartVal = chartTrack.val();

    sortTrack.empty().append('<option value="">Выберите ID свиньи</option>');
    chartTrack.empty().append('<option value="">Все ID</option>');

    Array.from(trackIds).sort((a, b) => a - b).forEach(track => {
      sortTrack.append(`<option value="${track}">${track}</option>`);
      chartTrack.append(`<option value="${track}">${track}</option>`);
    });

    sortTrack.val(currentSortVal || '');
    chartTrack.val(currentChartVal || '');
  }

  function showAlert(type, message) {
    const alertHtml = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
                        ${message}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                      </div>`;
    $("#alert-area").html(alertHtml);

    // Push-уведомление
    if (Notification.permission === 'granted') {
      new Notification(message);
    } else if (Notification.permission !== 'denied') {
      Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
          new Notification(message);
        }
      });
    }
  }

  function checkVideoStream() {
    if (Date.now() - lastFrameTime > frameTimeout) {
      $('#videoFeed').hide();
      $('#videoError').show();
    } else {
      $('#videoFeed').show();
      $('#videoError').hide();
    }
  }

  socket.on('connect', function() {
    clientId = socket.id;
    console.log('Connected with client_id:', clientId);
    loadInitialData();
    socket.emit('set_video_source', { source: $('#source').val() });
    setInterval(checkVideoStream, 1000);
  });

  socket.on('error', function(data) {
    console.error('SocketIO error:', data.message);
    showAlert('danger', data.message);
    $('#videoError').show();
  });

  socket.on('video_frame', function(msg) {
    try {
      $('#videoFeed').attr('src', 'data:image/jpeg;base64,' + msg.data);
      lastFrameTime = Date.now();
      checkVideoStream();
    } catch (e) {
      console.error('Error setting video frame:', e);
      $('#videoError').show();
    }
  });

  socket.on('source_changed', function(data) {
    console.log('Video source changed:', data.message);
    showAlert('success', data.message);
    $('#videoFeed').attr('src', '');
    $('#videoError').hide();
    lastFrameTime = 0;
    resetStats();
    loadLogs();
    loadChartData();
    startChartUpdates();
  });

  socket.on('state_added', function(data) {
    console.log('State added:', data.message);
    showAlert('success', data.message);
    loadStates(true);
  });

  socket.on('state_updated', function(data) {
    console.log('State updated:', data.message);
    showAlert('success', data.message);
    loadStates(true);
  });

  socket.on('state_deleted', function(data) {
    console.log('State deleted:', data.message);
    showAlert('success', data.message);
    loadStates(true);
  });

  socket.on('log_update', function(data) {
    try {
      updateLogTable(data.logs);
    } catch (e) {
      console.error('Error updating log table:', e);
    }
  });

  socket.on('chart_update', function(data) {
    try {
      stateCounts = data.state_counts || {};
      stateDurations = data.state_durations || {};
      const selectedTrack = $('#chartTrackFilter').val();
      initChart(selectedTrack);
      initDurationChart(selectedTrack);
      updateKPI();
    } catch (e) {
      console.error('Error updating charts:', e);
    }
  });

  socket.on('training_log', function(data) {
    const logDiv = $('#trainingLog');
    logDiv.append(`<p>${data.log}</p>`);
    logDiv.scrollTop(logDiv[0].scrollHeight);
  });

  socket.on('training_warning', function(data) {
    showAlert('warning', data.message);
    $('#trainModal').modal('hide');
  });

  socket.on('training_complete', function(data) {
    showAlert(data.message === 'OK' ? 'success' : 'danger', data.message === 'OK' ? 'Обучение завершено успешно' : 'Ошибка обучения');
    $('#trainModal').modal('hide');
    $('#trainingLog').empty();
  });

  $('#filterTime, #filterTrack, #sortTrack').on('input change', function() {
    loadLogs();
  });

  $('#chartTrackFilter').on('change', function() {
    const selectedTrack = $(this).val();
    initChart(selectedTrack);
    initDurationChart(selectedTrack);
  });

  $("#videoSourceForm").on("submit", function(e) {
    e.preventDefault();
    if (!clientId) {
      showAlert('danger', 'Клиент не подключен, попробуйте перезагрузить страницу');
      return;
    }
    const source = $('#source').val();
    if (chartUpdateInterval) {
      clearInterval(chartUpdateInterval);
    }
    socket.emit('set_video_source', { source: source });
  });

  $('#stateModal').on('show.bs.modal', function() {
    loadStates();
  });

  function loadStates(forceReload = false) {
    if (cachedStates && !forceReload) {
      renderStates(cachedStates);
      return;
    }
    $.ajax({
      url: '/get_states',
      method: 'GET',
      success: function(data) {
        cachedStates = data.states;
        renderStates(cachedStates);
      },
      error: function(xhr) {
        showAlert('danger', 'Ошибка загрузки состояний: ' + (xhr.responseJSON?.error || 'Неизвестная ошибка'));
      }
    });
  }

  function renderStates(states) {
    const tbody = $('#stateTable tbody');
    tbody.empty();
    states.forEach(function(state) {
      const isActive = state.code === currentStateCode ? 'table-active' : '';
      const row = `<tr class="${isActive}">
                    <td>${state.code}</td>
                    <td>${state.description}</td>
                    <td>
                      <button class="btn btn-sm btn-primary manage-dataset me-1" data-code="${state.code}">Управление датасетом</button>
                      <button class="btn btn-sm btn-warning edit-state me-1" data-code="${state.code}" data-description="${state.description}">Редактировать</button>
                      <button class="btn btn-sm btn-danger delete-state" data-code="${state.code}">Удалить</button>
                    </td>
                  </tr>`;
      tbody.append(row);
    });
    $('#stateModalBody').html(`
      <button class="btn btn-success mb-3" id="addStateBtn">Добавить состояние</button>
      <table class="table table-striped" id="stateTable">
        <thead>
          <tr>
            <th>Код</th>
            <th>Описание</th>
            <th>Действия</th>
          </tr>
        </thead>
        <tbody>${tbody.html()}</tbody>
      </table>
    `);
  }

  function loadVideos(stateCode, page = 1) {
    currentStateCode = stateCode;
    currentPage = page;
    $.ajax({
      url: `/get_videos/${stateCode}?page=${page}`,
      method: 'GET',
      success: function(data) {
        const tbody = $('#videoTable tbody');
        tbody.empty();
        data.files.forEach(function(file) {
          const fileType = file.toLowerCase().endsWith('.mp4') ? 'Видео' : 'Изображение';
          const row = `<tr>
                        <td>${file}</td>
                        <td>${fileType}</td>
                        <td>
                          <button class="btn btn-sm btn-info preview-file me-1" data-filename="${file}" data-type="${fileType.toLowerCase()}">Просмотр</button>
                          <button class="btn btn-sm btn-danger delete-file" data-filename="${file}">Удалить</button>
                        </td>
                      </tr>`;
          tbody.append(row);
        });
        $('#datasetModalLabel').text(`Управление датасетом: ${stateCode}`);
        totalPages = data.total_pages;
        currentPage = data.current_page;
        renderPagination();
      },
      error: function(xhr) {
        showAlert('danger', 'Ошибка загрузки файлов: ' + (xhr.responseJSON?.error || 'Неизвестная ошибка'));
      }
    });
  }

  function renderPagination() {
    const pagination = $('#pagination');
    pagination.empty();
    let startPage = Math.max(1, currentPage - 1);
    let endPage = Math.min(totalPages, startPage + 2);
    if (endPage - startPage < 2) {
      startPage = Math.max(1, endPage - 2);
    }

    pagination.append(
      `<li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${currentPage - 1}">Предыдущая</a>
      </li>`
    );

    for (let i = startPage; i <= endPage; i++) {
      pagination.append(
        `<li class="page-item ${i === currentPage ? 'active' : ''}">
          <a class="page-link" href="#" data-page="${i}">${i}</a>
        </li>`
      );
    }

    pagination.append(
      `<li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${currentPage + 1}">Следующая</a>
      </li>`
    );
  }

  $(document).on('click', '.page-link', function(e) {
    e.preventDefault();
    const page = $(this).data('page');
    if (page && page >= 1 && page <= totalPages) {
      loadVideos(currentStateCode, page);
    }
  });

  $(document).on('click', '#addStateBtn', function() {
    const modalBody = $('#stateModalBody');
    modalBody.html(`
      <form id="stateForm">
        <div class="mb-3">
          <label for="stateCode" class="form-label">Код состояния</label>
          <input type="text" class="form-control" id="stateCode" required>
        </div>
        <div class="mb-3">
          <label for="stateDescription" class="form-label">Описание</label>
          <input type="text" class="form-control" id="stateDescription" required>
        </div>
        <button type="submit" class="btn btn-primary">Сохранить</button>
        <button type="button" class="btn btn-secondary" id="cancelStateForm">Отмена</button>
      </form>
    `);
  });

  $(document).on('click', '#cancelStateForm', function() {
    loadStates();
  });

  $(document).on('submit', '#stateForm', function(e) {
    e.preventDefault();
    const code = $('#stateCode').val().trim();
    const description = $('#stateDescription').val().trim();
    const oldCode = $('#oldStateCode').val();
    if (oldCode) {
      socket.emit('update_state', { old_code: oldCode, new_code: code, description: description });
    } else {
      socket.emit('add_state', { code: code, description: description });
    }
  });

  $(document).on('click', '.edit-state', function() {
    const code = $(this).data('code');
    const description = $(this).data('description');
    const modalBody = $('#stateModalBody');
    modalBody.html(`
      <form id="stateForm">
        <div class="mb-3">
          <label for="stateCode" class="form-label">Код состояния</label>
          <input type="text" class="form-control" id="stateCode" value="${code}" required>
        </div>
        <div class="mb-3">
          <label for="stateDescription" class="form-label">Описание</label>
          <input type="text" class="form-control" id="stateDescription" value="${description}" required>
        </div>
        <input type="hidden" id="oldStateCode" value="${code}">
        <button type="submit" class="btn btn-primary">Сохранить</button>
        <button type="button" class="btn btn-secondary" id="cancelStateForm">Отмена</button>
      </form>
    `);
  });

  $(document).on('click', '.delete-state', function() {
    const code = $(this).data('code');
    if (confirm(`Удалить состояние ${code}?`)) {
      socket.emit('delete_state', { code: code });
    }
  });

  $(document).on('click', '.manage-dataset', function() {
    const stateCode = $(this).data('code');
    loadVideos(stateCode, 1);
    $('#stateModal').modal('hide');
    $('#datasetModal').modal('show');
  });

  $('#datasetModal').on('show.bs.modal', function() {
    if (currentStateCode) {
      loadVideos(currentStateCode, currentPage);
    }
  });

  $('#backToStateModal').on('click', function() {
    $('#datasetModal').modal('hide');
    $('#stateModal').modal('show');
  });

  $(document).on('click', '.preview-file', function() {
    const filename = $(this).data('filename');
    const fileType = $(this).data('type');
    const fileUrl = `/train/dataset/${currentStateCode}/${filename}`;
    if (fileType === 'видео') {
      $('#previewVideo').attr('src', fileUrl).show();
      $('#previewImage').hide();
    } else {
      $('#previewImage').attr('src', fileUrl).show();
      $('#previewVideo').hide();
    }
    $('#previewModalLabel').text(`Предпросмотр: ${filename}`);
    $('#datasetModal').modal('hide');
    $('#previewModal').modal('show');
  });

  $('#backToDatasetModal').on('click', function() {
    $('#previewModal').modal('hide');
    $('#datasetModal').modal('show');
  });

  $('#previewImage').on('click', function() {
    const src = $(this).attr('src');
    $('#enlargedImage').attr('src', src);
    $('#imagePreviewModalLabel').text('Увеличенное изображение');
    $('#previewModal').modal('hide');
    $('#imagePreviewModal').modal('show');
  });

  $('#imagePreviewModal').on('hidden.bs.modal', function() {
    $('#previewModal').modal('show');
  });

  $('#uploadVideoForm').on('submit', function(e) {
    e.preventDefault();
    const fileInput = $('#videoFile')[0];
    if (!fileInput.files.length) {
      showAlert('danger', 'Выберите файл для загрузки');
      return;
    }
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('video', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/upload_video/${currentStateCode}`, true);

    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        $('#uploadProgress').show();
        $('#uploadProgress .progress-bar').css('width', percent + '%').attr('aria-valuenow', percent);
      }
    };

    xhr.onload = function() {
      $('#uploadProgress').hide();
      if (xhr.status === 200) {
        showAlert('success', JSON.parse(xhr.responseText).message);
        loadVideos(currentStateCode, 1);
        $('#uploadVideoForm')[0].reset();
      } else {
        showAlert('danger', JSON.parse(xhr.responseText).error);
      }
    };

    xhr.onerror = function() {
      $('#uploadProgress').hide();
      showAlert('danger', 'Ошибка при загрузке файла');
    };

    xhr.send(formData);
  });

  $(document).on('click', '.delete-file', function() {
    const filename = $(this).data('filename');
    if (confirm(`Удалить файл ${filename}?`)) {
      $.ajax({
        url: `/delete_video/${currentStateCode}`,
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ filename: filename }),
        success: function(response) {
          showAlert('success', response.message);
          loadVideos(currentStateCode, currentPage);
        },
        error: function(xhr) {
          showAlert('danger', xhr.responseJSON.error);
        }
      });
    }
  });

  $('#startTrainingBtn').on('click', function() {
    $('#trainingLog').empty();
    $.ajax({
      url: '/train_model',
      method: 'POST',
      success: function(response) {
        showAlert('success', response.message);
      },
      error: function(xhr) {
        showAlert('danger', xhr.responseJSON.error);
      }
    });
  });

  loadInitialData();
});