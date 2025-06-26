// src/router/index.js
import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import LoginView from "../views/LoginView.vue";
import RegisterView from "../views/RegisterView.vue";
import HistoryView from "../views/HistoryView.vue";

const routes = [
  {
    path: "/",
    name: "home",
    component: HomeView,
  },
  {
    path: "/login",
    name: "login",
    component: LoginView,
  },
  {
    path: "/register",
    name: "register",
    component: RegisterView,
  },
  {
    path: "/history",
    name: "history",
    component: HistoryView,
   
  },
];

const router = createRouter({
  // Ubah baris ini:
  history: createWebHistory(import.meta.env.BASE_URL), // <-- PERBAIKAN DI SINI
  routes,
});

// Navigation Guard tetap sama
router.beforeEach((to, from, next) => {
  if (to.meta.requiresAuth && !localStorage.getItem("access_token")) {
    alert("Anda harus login untuk mengakses halaman ini.");
    next("/login");
  } else {
    next();
  }
});

export default router;
