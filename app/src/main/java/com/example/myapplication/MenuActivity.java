package com.example.myapplication;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.os.Build.VERSION.SDK_INT;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.navigation.NavController;

public class MenuActivity extends AppCompatActivity {
    private NavController navController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

    }

    public void recorderPage(View view) {
        Intent i = new Intent(this,Recorder.class);
        startActivity(i);
    }

    public void btnPastSpeech(View view) {
        /*
        Should display the page of the past recordings with a small audio player at the
        bottom of the screen. Please use recylerView.
        */
        Intent intent= new Intent(MenuActivity.this, HistoryActivity.class);
        startActivity(intent);

    }
}