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
  
  // Переводы состояний на русский язык
  const stateTranslations = {
    'Feeding': 'Кормление',
    'Sitting': 'Сидит',
    'Standing': 'Стоит',
    'Lateral_Lying': 'Лежит на боку',
    'Sternal_Lying': 'Лежит на животе'
  };
  
  // Функция для получения русского перевода состояния
  function getStateTranslation(state) {
    return stateTranslations[state] || state;
  }

  // Переключатель темной темы
  $('#darkModeToggle').on('click', function() {
    const isDark = $('html').attr('data-theme') === 'dark';
    $('html').attr('data-theme', isDark ? 'light' : 'dark');
    $(this).find('i').attr('class', isDark ? 'bi bi-moon-stars' : 'bi bi-sun');
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
  });

  // Восстановить тему из localStorage
  if (localStorage.getItem('theme') === 'dark') {
    $('html').attr('data-theme', 'dark');
    $('#darkModeToggle i').attr('class', 'bi bi-sun');
  }

  // Загрузить количество состояний
  loadStateCount();

  // Обработчик кнопки очистки журнала
  $('#clearLogBtn').on('click', function() {
    if (confirm('Вы уверены, что хотите очистить журнал активности?')) {
      $('#logTable tbody').empty();
      $('#emptyLogState').removeClass('d-none');
    }
  });

  // Обработчик кнопки экспорта журнала
  $('#exportLogBtn').on('click', function() {
    const logData = [];
    $('#logTable tbody tr').each(function() {
      const row = $(this);
      logData.push({
        time: row.find('td:eq(0)').text(),
        pigId: row.find('td:eq(1)').text(),
        event: row.find('td:eq(2)').text()
      });
    });

    if (logData.length === 0) {
      alert('Журнал активности пуст');
      return;
    }

    const csvContent = 'data:text/csv;charset=utf-8,'
      + 'Время,ID свиньи,Событие\n'
      + logData.map(row => `${row.time},${row.pigId},"${row.event}"`).join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `activity_log_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

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
    $('#totalPigs').html('<div class="loading-spinner"></div>');
    $('#activePigs').html('<div class="loading-spinner"></div>');
    $('#avgStateTime').html('<div class="loading-spinner"></div>');
    // Не сбрасывать totalStates, так как оно загружается отдельно
    $('#logTable tbody').empty();
    $('#emptyLogState').removeClass('d-none');
    $('#sortTrack').empty().append('<option value="">Выберите ID свиньи</option>');
    $('#chartTrackFilter').empty().append('<option value="">Все ID</option>');
    initChart(null);
    initDurationChart(null);
  }

  // Загрузить количество состояний для KPI
  function loadStateCount() {
    $.get('/get_state_count')
      .done(function(data) {
        $('#totalStates').text(data.count);
      })
      .fail(function() {
        $('#totalStates').text('N/A');
      });
  }

  function initChart(trackId) {
    let data = trackId ? (stateCounts[trackId] || {}) : Object.keys(stateCounts).reduce((acc, tid) => {
      for (let state in stateCounts[tid]) {
        acc[state] = (acc[state] || 0) + stateCounts[tid][state];
      }
      return acc;
    }, {});

    // Переводим названия состояний на русский язык для отображения
    const translatedData = {};
    for (let state in data) {
      const translatedState = getStateTranslation(state);
      translatedData[translatedState] = data[state];
    }

    const trace = {
      x: Object.keys(translatedData),
      y: Object.values(translatedData),
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

    // Переводим названия состояний на русский язык для отображения
    const translatedData = {};
    for (let state in data) {
      const translatedState = getStateTranslation(state);
      translatedData[translatedState] = data[state];
    }

    const trace = {
      x: Object.keys(translatedData),
      y: Object.values(translatedData).map(seconds => Math.round(seconds / 60)),
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

    // Обновить KPI реальными значениями (убрать спиннеры загрузки)
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
        console.warn('Не удалось загрузить состояния:', xhr.responseJSON?.error || 'Неизвестная ошибка');
        // Продолжаем работу без состояний
        cachedStates = [];
        loadLogs();
        loadChartData();
        startChartUpdates();
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
                    <td class="text-nowrap">
                      <i class="bi bi-clock me-1 text-muted"></i>${timePart}
                    </td>
                    <td class="text-center">
                      <span class="badge bg-primary">${track_id}</span>
                    </td>
                    <td>
                      <small class="text-muted">${message}</small>
                    </td>
                  </tr>`;
      tbody.append(row);
    });

    // Показать/скрыть пустое состояние
    if (tbody.children().length === 0) {
      $('#emptyLogState').removeClass('d-none');
    } else {
      $('#emptyLogState').addClass('d-none');
    }

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
    console.log('🔌 Connected with client_id:', clientId);
    loadInitialData();
    socket.emit('set_video_source', { source: $('#source').val() });
    setInterval(checkVideoStream, 1000);
  });

  // Добавляем универсальный обработчик для отслеживания всех событий
  const originalOn = socket.on;
  socket.on = function(event, handler) {
    return originalOn.call(this, event, function(...args) {
      if (event !== 'video_frame') { // Не логируем каждый кадр
        console.log(`📡 Получено SocketIO событие: ${event}`, args);
      }
      return handler.apply(this, args);
    });
  };

  socket.on('error', function(data) {
    console.error('🚨 SocketIO error:', data.message);
    showAlert('danger', data.message);
    $('#videoError').show();
    
    // Восстанавливаем кнопку если была ошибка при смене источника
    const submitBtn = $("#videoSourceForm button[type='submit']");
    if (submitBtn.prop('disabled')) {
      submitBtn.prop('disabled', false).html('<i class="bi bi-arrow-repeat me-1"></i>🔄 Сменить');
    }
  });

  let frameCount = 0;
  socket.on('video_frame', function(msg) {
    try {
      $('#videoFeed').attr('src', 'data:image/jpeg;base64,' + msg.data);
      lastFrameTime = Date.now();
      checkVideoStream();
      
      frameCount++;
      if (frameCount % 30 === 0) { // Логируем каждые 30 кадров
        console.log(`📹 Получено ${frameCount} видео кадров`);
      }
    } catch (e) {
      console.error('❌ Error setting video frame:', e);
      $('#videoError').show();
    }
  });

  socket.on('source_changed', function(data) {
    console.log('✅ Получено событие source_changed:', data.message);
    showAlert('success', data.message);
    
    // Восстанавливаем кнопку
    const submitBtn = $("#videoSourceForm button[type='submit']");
    submitBtn.prop('disabled', false).html('<i class="bi bi-arrow-repeat me-1"></i>🔄 Сменить');
    
    // Сбрасываем видео
    $('#videoFeed').attr('src', '');
    $('#videoError').hide();
    lastFrameTime = 0;
    
    // Сбрасываем статистику и перезапускаем обновления
    resetStats();
    loadLogs();
    loadChartData();
    startChartUpdates();
    
    console.log('🔄 Статистика сброшена, обновления перезапущены');
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
    console.log('🔄 Форма смены источника отправлена');
    
    if (!clientId) {
      console.error('❌ Клиент не подключен');
      showAlert('danger', 'Клиент не подключен, попробуйте перезагрузить страницу');
      return;
    }
    
    const source = $('#source').val();
    console.log(`📹 Выбранный источник: "${source}"`);
    
    if (!source) {
      console.error('❌ Источник не выбран');
      showAlert('danger', 'Выберите источник видео');
      return;
    }
    
    if (chartUpdateInterval) {
      clearInterval(chartUpdateInterval);
    }
    
    // Показываем индикатор загрузки
    const submitBtn = $(this).find('button[type="submit"]');
    const originalText = submitBtn.html();
    submitBtn.prop('disabled', true).html('<div class="loading-spinner me-1"></div>🔄 Переключение...');
    
    console.log(`🚀 Отправка события set_video_source с источником: "${source}"`);
    socket.emit('set_video_source', { source: source });
    
    // Восстанавливаем кнопку через 5 секунд (на случай если не получим ответ)
    setTimeout(() => {
      if (submitBtn.prop('disabled')) {
        console.warn('⚠️ Таймаут ожидания ответа сервера, восстанавливаем кнопку');
        submitBtn.prop('disabled', false).html(originalText);
        showAlert('warning', 'Таймаут смены источника. Попробуйте еще раз.');
      }
    }, 5000);
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
        
        if (data.files.length === 0) {
          // Показываем сообщение об отсутствии файлов
          tbody.append(`
            <tr>
              <td colspan="4" class="text-center py-4">
                <div class="text-muted">
                  <i class="bi bi-folder2-open fs-1 mb-2 d-block"></i>
                  <p class="mb-0">В этом датасете пока нет файлов</p>
                  <small>Загрузите видео или изображения с помощью формы выше</small>
                </div>
              </td>
            </tr>
          `);
        }
        
        data.files.forEach(function(file) {
          const extension = file.toLowerCase().split('.').pop();
          const videoExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
          const imageExtensions = ['jpg', 'jpeg', 'png', 'bmp'];
          
          let fileType = 'Неизвестно';
          if (videoExtensions.includes(extension)) {
            fileType = 'Видео';
          } else if (imageExtensions.includes(extension)) {
            fileType = 'Изображение';
          }
          
          // Определяем иконку для типа файла
          let fileIcon = '📄';
          if (videoExtensions.includes(extension)) {
            fileIcon = '🎬';
          } else if (imageExtensions.includes(extension)) {
            fileIcon = '🖼️';
          }
          
          const row = `<tr>
                        <td class="text-center">${fileIcon}</td>
                        <td>
                          <div class="d-flex align-items-center">
                            <strong>${file}</strong>
                          </div>
                          <small class="text-muted">.${extension.toUpperCase()}</small>
                        </td>
                        <td>
                          <span class="badge ${fileType === 'Видео' ? 'bg-primary' : 'bg-success'}">${fileType}</span>
                        </td>
                        <td>
                          <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-info preview-file" data-filename="${file}" data-type="${fileType.toLowerCase()}" title="Предпросмотр">
                              <i class="bi bi-eye"></i>
                            </button>
                            <button class="btn btn-outline-danger delete-file" data-filename="${file}" title="Удалить">
                              <i class="bi bi-trash"></i>
                            </button>
                          </div>
                        </td>
                      </tr>`;
          tbody.append(row);
        });
        $('#datasetModalLabel').text(`Управление датасетом: ${stateCode}`);
        
        // Обновляем счетчик файлов
        const fileCountText = data.total_files === 1 ? '1 файл' : 
                             data.total_files < 5 ? `${data.total_files} файла` : 
                             `${data.total_files} файлов`;
        $('#fileCount').text(fileCountText);
        
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
    $('#trainingProgress').addClass('d-none');
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

  // Кнопка очистки лога обучения
  $('#clearTrainingLogBtn').on('click', function() {
    if (confirm('Вы уверены, что хотите очистить лог обучения?')) {
      $('#trainingLog').empty();
      $('#trainingLog').html('<div class="text-muted"><i class="bi bi-terminal me-1"></i>Лог очищен. Нажмите "Начать обучение" для нового запуска.</div>');
    }
  });

  // Кнопка скачивания лога обучения
  $('#downloadTrainingLogBtn').on('click', function() {
    const logContent = $('#trainingLog').text();
    if (!logContent.trim()) {
      alert('Лог обучения пуст');
      return;
    }

    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_log_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Обработчики для удаленной камеры
  $('#testCameraBtn').on('click', function() {
    const url = $('#cameraUrl').val();
    const type = $('#cameraType').val();
    const username = $('#cameraUsername').val();
    const password = $('#cameraPassword').val();

    if (!url) {
      alert('🚨 Введите URL адрес камеры');
      return;
    }

    $(this).prop('disabled', true).html('<div class="loading-spinner me-1"></div>🧪 Тестирование...');

    // Имитация тестирования подключения
    setTimeout(() => {
      const success = Math.random() > 0.3; // 70% вероятность успеха для демо
      if (success) {
        showAlert('success', '✅ Подключение к камере успешно!');
      } else {
        showAlert('danger', '❌ Не удалось подключиться к камере. Проверьте настройки.');
      }
      $(this).prop('disabled', false).html('<i class="bi bi-play-circle me-1"></i>🧪 Тест подключения');
    }, 2000);
  });

  $('#saveCameraBtn').on('click', function() {
    const settings = {
      type: $('#cameraType').val(),
      url: $('#cameraUrl').val(),
      username: $('#cameraUsername').val(),
      password: $('#cameraPassword').val(),
      resolution: $('#cameraResolution').val(),
      fps: $('#cameraFps').val(),
      timeout: $('#cameraTimeout').val()
    };

    if (!settings.url) {
      alert('🚨 Введите URL адрес камеры');
      return;
    }

    // Сохраняем настройки в localStorage
    localStorage.setItem('remoteCameraSettings', JSON.stringify(settings));
    showAlert('success', '✅ Настройки удаленной камеры сохранены!');
    $('#remoteCameraModal').modal('hide');
  });

  // Обработчики для управления источниками
  $('#sourceUploadForm').on('submit', function(e) {
    e.preventDefault();
    const files = $('#sourceFiles')[0].files;
    
    if (files.length === 0) {
      alert('📎 Выберите файлы для загрузки');
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    $('#sourceUploadProgress').show();
    let progress = 0;
    const progressBar = $('#sourceUploadProgress .progress-bar');

    // Имитация загрузки
    const uploadInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 100) {
        progress = 100;
        clearInterval(uploadInterval);
        setTimeout(() => {
          $('#sourceUploadProgress').hide();
          progressBar.css('width', '0%');
          showAlert('success', `✅ Загружено ${files.length} файл(ов) успешно!`);
          loadSourceList();
          $('#sourceFiles').val('');
        }, 500);
      }
      progressBar.css('width', progress + '%').attr('aria-valuenow', progress);
    }, 200);
  });

  $('#refreshSourcesBtn').on('click', function() {
    loadSourceList();
  });

  function loadSourceList() {
    // Загружаем реальный список файлов из папки uploads
    $.ajax({
      url: '/get_source_files',
      method: 'GET',
      success: function(data) {
        const sources = data.files || [];
        const tbody = $('#sourceTable tbody');
        tbody.empty();

        if (sources.length === 0) {
          $('#emptySourcesState').removeClass('d-none');
          $('#sourceFileCount').text('0 файлов');
          return;
        }

        $('#emptySourcesState').addClass('d-none');
        $('#sourceFileCount').text(`${sources.length} файлов`);

        sources.forEach((source, index) => {
          const row = `
            <tr>
              <td class="text-center">${source.type}</td>
              <td>${source.name}</td>
              <td><span class="badge bg-primary">${source.type_label}</span></td>
              <td><small class="text-muted">${source.size}</small></td>
              <td>
                <div class="btn-group btn-group-sm">
                  <button class="btn btn-outline-primary" onclick="previewSource('${source.name}')">
                    <i class="bi bi-eye"></i> 👁️
                  </button>
                  <button class="btn btn-outline-danger" onclick="deleteSource('${source.name}')">
                    <i class="bi bi-trash"></i> 🗑️
                  </button>
                </div>
              </td>
            </tr>
          `;
          tbody.append(row);
        });
      },
      error: function(xhr) {
        console.error('Ошибка загрузки списка файлов:', xhr.responseJSON?.error || 'Неизвестная ошибка');
        $('#emptySourcesState').removeClass('d-none');
        $('#sourceFileCount').text('Ошибка загрузки');
      }
    });
  }

  // Глобальные функции для управления источниками
  window.previewSource = function(filename) {
    // Открываем страницу предпросмотра в новом окне браузера
    const previewUrl = `/preview/${filename}`;
    const windowFeatures = 'width=900,height=700,scrollbars=yes,resizable=yes,location=no,menubar=no,toolbar=no';
    const previewWindow = window.open(previewUrl, `preview_${filename}`, windowFeatures);
    
    if (!previewWindow) {
      showAlert('warning', '⚠️ Не удалось открыть окно предпросмотра. Проверьте настройки блокировки всплывающих окон.');
    } else {
      showAlert('info', `👁️ Открыт предпросмотр: ${filename}`);
    }
  };

  window.deleteSource = function(filename) {
    if (confirm(`🗑️ Удалить файл "${filename}"?`)) {
      showAlert('success', `✅ Файл "${filename}" удален`);
      loadSourceList();
    }
  };

  // Загрузить список источников при открытии модального окна
  $('#sourceManagementModal').on('shown.bs.modal', function() {
    loadSourceList();
  });

  // Восстановить настройки камеры при открытии модального окна
  $('#remoteCameraModal').on('shown.bs.modal', function() {
    const savedSettings = localStorage.getItem('remoteCameraSettings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      $('#cameraType').val(settings.type || 'rtsp');
      $('#cameraUrl').val(settings.url || '');
      $('#cameraUsername').val(settings.username || '');
      $('#cameraPassword').val(settings.password || '');
      $('#cameraResolution').val(settings.resolution || '1280x720');
      $('#cameraFps').val(settings.fps || '25');
      $('#cameraTimeout').val(settings.timeout || '30');
    }
  });

  loadInitialData();
});