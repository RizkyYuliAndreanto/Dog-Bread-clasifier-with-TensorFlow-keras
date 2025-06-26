<template>
    <div class="home-view">
      <HeroSection />
      <section class="prediction-section container">
        <h2 class="section-title">Unggah Gambar Anjing untuk Prediksi</h2>
        
        <!-- Perubahan di sini: Tidak lagi menyembunyikan prediction-card berdasarkan isAuthenticated -->
        <div class="prediction-card">
          <input type="file" @change="onFileSelected" accept="image/*" class="file-input" />
          
          <div v-if="selectedFile">
            <p class="file-info">File yang dipilih: <strong>{{ selectedFile.name }}</strong></p>
            <img v-if="selectedFilePreview" :src="selectedFilePreview" alt="Preview Gambar" class="image-preview">
          </div>
 
          <button
            @click="uploadFile"
            :disabled="!selectedFile || uploading"
            class="btn btn-primary btn-block"
          >
            {{ uploading ? 'Mengunggah & Memprediksi...' : 'Prediksi Sekarang' }}
          </button>
 
          <div v-if="predictionResult" class="prediction-result">
            <h3>Hasil Prediksi:</h3>
            <p>Ras: <strong>{{ predictionResult.predicted_breed }}</strong></p>
            <p>Keyakinan: <strong>{{ predictionResult.confidence }}</strong></p>
            <img
            v-if="predictionResult.image_url"
            :src="backendBaseUrl + predictionResult.image_url"
            alt="Gambar Terprediksi"
            class="predicted-image"
            >
          </div>
          <p v-if="uploadError" class="alert-message error">{{ uploadError }}</p>
        </div>
      </section>
    </div>
</template>
 
<script>
import apiClient from '@/plugins/axios';
import HeroSection from '@/components/HeroSection.vue';
 
export default {
    name: 'HomeView',
    components: {
      HeroSection
    },
    data() {
      return {
        selectedFile: null,
        selectedFilePreview: null,
        predictionResult: null,
        uploadError: '',
        uploading: false,
        isAuthenticated: false, // Tetap ada untuk keperluan lain (misal: tampilan Navbar)
        backendBaseUrl: 'http://127.0.0.1:5000'
      };
    },
    created() {
      this.checkAuthStatus();
      window.addEventListener('storage', this.checkAuthStatus);
    },
    beforeUnmount() {
      window.removeEventListener('storage', this.checkAuthStatus);
    },
    methods: {
      checkAuthStatus() {
        // Ini tetap penting untuk UI lain yang bergantung pada status login,
        // seperti tombol Login/Logout di Navbar.
        this.isAuthenticated = !!localStorage.getItem('access_token');
      },
      onFileSelected(event) {
        const file = event.target.files[0];
        this.selectedFile = file;
        this.predictionResult = null;
        this.uploadError = '';
        
        if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
            this.selectedFilePreview = e.target.result;
          };
          reader.readAsDataURL(file);
        } else {
          this.selectedFilePreview = null;
        }
      },
      async uploadFile() {
        if (!this.selectedFile) {
          this.uploadError = 'Pilih file gambar terlebih dahulu.';
          return;
        }
 
        // --- PERUBAHAN DI SINI: Hapus pengecekan `isAuthenticated` untuk prediksi ---
        // if (!this.isAuthenticated) {
        //   this.uploadError = 'Anda harus login untuk memprediksi.';
        //   return;
        // }
 
        // Token tetap diambil, tapi tidak wajib ada untuk rute /predict
        // Ini akan secara otomatis ditambahkan oleh interceptor Axios jika ada
        // dan diabaikan oleh backend karena rute tidak memerlukan JWT.
        // const token = localStorage.getItem('access_token'); 
        // if (!token) { // Hapus pengecekan ini jika ingin prediksi tanpa login
        //    this.uploadError = 'Token tidak ditemukan. Silakan login ulang.';
        //    this.$router.push('/login');
        //    return;
        // }
 
        const formData = new FormData();
        formData.append('file', this.selectedFile);
 
        this.uploading = true;
        this.predictionResult = null;
        this.uploadError = '';
        
        try {
          // Axios interceptor akan otomatis menambahkan header Authorization jika token ada.
          // Kita tidak perlu menambahkannya secara eksplisit di sini lagi.
          const response = await apiClient.post('/predict', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
              // 'Authorization': `Bearer ${token}` // Hapus ini karena interceptor sudah menangani
            },
            withCredentials: true // Tetap pertahankan ini untuk CORS jika diperlukan
          });
          
          if (response.status === 200) {
            this.predictionResult = response.data;
          } else {
            this.uploadError = response.data.error || 'Prediksi gagal dengan status tidak terduga.';
          }
 
        } catch (error) {
          this.predictionResult = null;
 
          if (error.response) {
            const status = error.response.status;
            // Hapus kondisi 401/403 untuk rute /predict, karena sekarang publik
            // Kondisi ini hanya berlaku untuk rute yang dilindungi seperti /history
            if (status === 400) {
              this.uploadError = error.response.data.error || 'Format gambar salah atau tidak ada file.';
            } else if (status === 500) {
              this.uploadError = error.response.data.error || 'Terjadi kesalahan internal server.';
            } else {
              this.uploadError = `Terjadi kesalahan. Status: ${status}`;
            }
          } else if (error.request) {
            this.uploadError = 'Tidak ada respons dari server. Periksa koneksi internet Anda atau pastikan server aktif.';
          } else {
            this.uploadError = `Kesalahan saat mengirim permintaan: ${error.message}`;
          }
          
          console.error('Error uploading file for prediction:', error);
        } finally {
          this.uploading = false;
        }
      }
    }
};
</script>
 
<style scoped>
/* Styling tetap sama seperti sebelumnya */
.prediction-section {
    padding: 40px 20px;
    text-align: center;
}
 
.section-title {
    font-size: 2.5em;
    margin-bottom: 30px;
    color: #333;
}
 
.prediction-card {
    background-color: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    max-width: 500px;
    margin: 0 auto;
}
 
.file-input {
    display: block;
    width: 100%;
    padding: 10px;
    margin-bottom: 20px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #f9f9f9;
}
 
.file-info {
    margin-bottom: 15px;
    font-size: 0.95em;
    color: #555;
}
 
.image-preview {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid #eee;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
 
.prediction-result {
    margin-top: 25px;
    padding: 15px;
    background-color: #e6ffe6;
    border: 1px solid #aaffaa;
    border-radius: 5px;
    text-align: left;
}
 
.prediction-result h3 {
    font-size: 1.3em;
    color: #28a745;
    margin-bottom: 10px;
    text-align: center;
}
 
.prediction-result p {
    font-size: 1.1em;
    font-weight: 500;
    margin-bottom: 5px;
}
 
.predicted-image {
    max-width: 100%;
    height: auto;
    margin-top: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.alert-message.error {
    color: red;
    margin-top: 15px;
}
</style>
